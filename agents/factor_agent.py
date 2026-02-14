"""
Factor Agent - 因子计算与单因子检验（注册表驱动版）

重构要点（融入 Ray 的设计）：
- 因子计算逻辑从 FactorAgent 中剥离到独立模块（factors/）
- 通过 FactorRegistry 自动发现和调用所有已注册因子
- 每个因子经过标准化的 validate + clean 质量检验流程
- 新增因子只需在 factors/ 下写一个文件，无需修改 FactorAgent
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_DIR, OUTPUT_DIR
from utils.helpers import save_parquet, load_parquet, ensure_dir
from core.agent_base import BaseAgent


class FactorAgent(BaseAgent):
    """因子Agent：注册表驱动的因子计算与检验"""

    agent_name = "FactorAgent"

    def __init__(self, daily_quotes: pd.DataFrame):
        super().__init__()
        self.df = daily_quotes.copy()
        self.df = self.df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self.factors = None

        import factors  # noqa: F401
        from factor_framework.registry import FactorRegistry
        self.registry = FactorRegistry()

        enabled = self.registry.list_factors(enabled_only=True)
        print(f"[FactorAgent] 初始化完成，已注册 {len(enabled)} 个启用因子")

    def compute_factors(self) -> pd.DataFrame:
        """通过注册表计算所有启用的因子"""
        cache_path = os.path.join(DATA_DIR, "factors_v2.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[FactorAgent] 因子数据已缓存，共 {len(cached)} 行")
            self.factors = cached
            return cached

        print("[FactorAgent] 正在计算因子...")
        df = self.df.copy()

        # 日收益率（非因子，但多个因子的计算基础）
        df["ret"] = df.groupby("ts_code")["close"].pct_change()

        # 通过注册表逐个计算因子
        enabled_names = self.registry.list_factors(enabled_only=True)
        for fname in enabled_names:
            instance = self.registry.create_instance(fname)

            ts = datetime.now()
            df[fname] = instance.compute(df)
            elapsed = (datetime.now() - ts).total_seconds()

            # 质量检验（Ray 的 validate 模式）
            quality = instance.validate(df[fname])
            if quality["inf_count"] > 0:
                print(f"  警告: {fname} 包含 {quality['inf_count']} 个 Inf 值")
            if quality["nan_ratio"] > 0.5:
                print(f"  警告: {fname} NaN 比例 {quality['nan_ratio']:.1%}，超过阈值")

            # 清洗（Inf→NaN, winsorize）
            df[fname] = instance.clean(df[fname])

            print(f"  {instance.metadata.display_name}({fname}): "
                  f"耗时 {elapsed:.1f}s, NaN={quality['nan_ratio']:.1%}")

        # 未来收益率（用于因子检验，非因子本身）
        df["fwd_ret_1m"] = df.groupby("ts_code")["close"].transform(
            lambda x: x.shift(-20) / x - 1
        )

        # 排除 IPO 期数据（上市前5个交易日，价格异常不参与因子评估）
        if "is_ipo_period" in df.columns:
            n_before_ipo = len(df)
            df = df[df["is_ipo_period"] == 0]
            print(f"  排除 IPO 期: {n_before_ipo - len(df)} 行")

        # 构建结果
        base_cols = ["ts_code", "trade_date", "close", "ret"]
        factor_cols = base_cols + enabled_names + ["fwd_ret_1m"]
        result = df[factor_cols].dropna(subset=[enabled_names[0], "fwd_ret_1m"])

        save_parquet(result, cache_path)
        self.factors = result
        print(f"[FactorAgent] 因子计算完成，共 {len(result)} 行")

        # 保存注册表元信息
        registry_path = os.path.join(OUTPUT_DIR, "factor_registry.json")
        ensure_dir(OUTPUT_DIR)
        self.registry.save_to_disk(registry_path)

        return result

    def single_factor_test(self) -> pd.DataFrame:
        """对每个启用因子进行 RankIC 检验，并回写评价指标到注册表"""
        if self.factors is None:
            self.compute_factors()

        enabled_names = self.registry.list_factors(enabled_only=True)

        print("[FactorAgent] 正在进行单因子检验...")
        results = []

        for fname in enabled_names:
            ic_series = self._calc_rank_ic(fname)
            if len(ic_series) == 0:
                continue

            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_pct = (ic_series > 0).mean()

            results.append({
                "因子名称": fname,
                "RankIC均值": round(ic_mean, 4),
                "RankIC标准差": round(ic_std, 4),
                "IR": round(ir, 4),
                "IC>0占比": round(ic_positive_pct, 4),
            })

            # 回写评价指标到注册表（Ray 的 lineage 模式）
            try:
                meta = self.registry.get_metadata(fname)
                meta.eval_ic_mean = round(ic_mean, 4)
                meta.eval_ir = round(ir, 4)
                self.registry.update_metadata(fname, meta)
            except KeyError:
                pass

            print(f"  {fname}: IC={ic_mean:.4f}, IR={ir:.4f}, IC>0占比={ic_positive_pct:.2%}")

        result_df = pd.DataFrame(results)
        ensure_dir(OUTPUT_DIR)
        result_df.to_csv(os.path.join(OUTPUT_DIR, "single_factor_test.csv"),
                         index=False, encoding="utf-8-sig")

        # 持久化更新后的注册表（含评价指标）
        registry_path = os.path.join(OUTPUT_DIR, "factor_registry.json")
        self.registry.save_to_disk(registry_path)

        # V3: 发出因子检验结果信号
        test_results = {}
        for _, row in result_df.iterrows():
            test_results[row["因子名称"]] = {
                "ic_mean": row["RankIC均值"],
                "ir": row["IR"],
            }
        self.emit("factors.tested", {"results": test_results})

        print("[FactorAgent] 单因子检验完成")
        return result_df

    def _calc_rank_ic(self, factor_name: str) -> pd.Series:
        """计算某个因子的逐期 RankIC"""
        df = self.factors.dropna(subset=[factor_name, "fwd_ret_1m"])

        df = df.copy()
        df["month"] = df["trade_date"].str[:6]
        month_ends = df.groupby("month")["trade_date"].max().values
        df_monthly = df[df["trade_date"].isin(month_ends)]

        ic_list = []
        for date, group in df_monthly.groupby("trade_date"):
            if len(group) < 30:
                continue
            corr, _ = stats.spearmanr(group[factor_name], group["fwd_ret_1m"])
            if not np.isnan(corr):
                ic_list.append({"trade_date": date, "ic": corr})

        return pd.DataFrame(ic_list)["ic"] if ic_list else pd.Series(dtype=float)

    def get_factor_scores(self, date: str) -> pd.DataFrame:
        """获取指定日期的因子截面数据（供 AlphaAgent 使用）"""
        if self.factors is None:
            self.compute_factors()

        snapshot = self.factors[self.factors["trade_date"] == date].copy()
        enabled_names = self.registry.list_factors(enabled_only=True)

        # 只保留有数据的因子，且截面 NaN 比例 < 50%
        available = []
        for f in enabled_names:
            if f in snapshot.columns:
                nan_ratio = snapshot[f].isna().mean()
                if snapshot[f].notna().sum() > 0 and nan_ratio < 0.5:
                    available.append(f)

        # 截面标准化（z-score）
        for fname in available:
            col = snapshot[fname]
            std = col.std()
            if std > 0:
                snapshot[fname] = (col - col.mean()) / std
            else:
                snapshot[fname] = 0

        return snapshot[["ts_code"] + available].dropna()


if __name__ == "__main__":
    quotes = load_parquet(os.path.join(DATA_DIR, "daily_quotes.parquet"))
    if quotes is not None:
        agent = FactorAgent(quotes)
        agent.compute_factors()
        agent.single_factor_test()
    else:
        print("请先运行 DataAgent 拉取数据")
