"""
Data Agent - 数据采集、清洗、本地缓存
混合数据源：Tushare daily(按日期) + AKShare index

V3 数据清洗链路：
1. 获取股票基本信息（名称、上市日期）→ 识别 ST
2. 拉取全市场行情
3. 清洗：过滤ST、标记IPO首日、标记涨跌停、计算日收益率
4. 输出带标记的干净数据
"""
import os
import json
import time
import tushare as ts
import akshare as ak
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    TUSHARE_TOKEN,
    PRICE_ADJ_MODE,
    STOCK_INFO_PROVIDER,
    DATA_DIR,
    START_DATE,
    END_DATE,
    API_DELAY,
)
from utils.helpers import save_parquet, load_parquet, ensure_dir
from core.agent_base import BaseAgent


class DataAgent(BaseAgent):
    """
    数据Agent（混合数据源 + 数据清洗）
    - Tushare daily(trade_date): 按日期拉全市场行情（免费可用）
    - Tushare stock_basic: 股票名称/上市日期（识别 ST）
    - AKShare: 指数日线
    """

    agent_name = "DataAgent"

    def __init__(self):
        super().__init__()
        self.price_adj_mode = PRICE_ADJ_MODE if PRICE_ADJ_MODE in {"qfq", "hfq", "none"} else "qfq"
        if self.price_adj_mode != PRICE_ADJ_MODE:
            print(f"[DataAgent] 警告: 非法 PRICE_ADJ_MODE={PRICE_ADJ_MODE}，已回退为 qfq。")
        self.stock_info_provider = (
            STOCK_INFO_PROVIDER if STOCK_INFO_PROVIDER in {"auto", "tushare", "akshare"} else "auto"
        )
        if self.stock_info_provider != STOCK_INFO_PROVIDER:
            print(f"[DataAgent] 警告: 非法 STOCK_INFO_PROVIDER={STOCK_INFO_PROVIDER}，已回退为 auto。")

        self.pro = None
        if TUSHARE_TOKEN:
            ts.set_token(TUSHARE_TOKEN)
            self.pro = ts.pro_api()
        else:
            print("[DataAgent] 警告: 未设置 TUSHARE_TOKEN，远程拉取已禁用，仅可使用本地缓存。")
        ensure_dir(DATA_DIR)
        print(
            f"[DataAgent] 初始化完成（PRICE_ADJ_MODE={self.price_adj_mode}, "
            f"STOCK_INFO_PROVIDER={self.stock_info_provider}）"
        )

    def _require_pro_api(self, action: str) -> None:
        """Ensure Tushare API is available before remote fetch."""
        if self.pro is None:
            raise RuntimeError(
                f"[DataAgent] 执行 {action} 需要有效的 TUSHARE_TOKEN。"
                "请先设置环境变量 TUSHARE_TOKEN，或提前准备本地缓存。"
            )

    @staticmethod
    def _load_cache_meta(meta_path: str) -> dict:
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_cache_meta(meta_path: str, data: dict) -> None:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_stock_info(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["ts_code", "name", "list_date", "delist_date", "list_status", "is_st"]
        out = df.copy()
        for col in required_cols:
            if col not in out.columns:
                out[col] = ""
        out["name"] = out["name"].fillna("").astype(str)
        out["list_date"] = out["list_date"].fillna("").astype(str)
        out["delist_date"] = out["delist_date"].fillna("").astype(str)
        out["list_status"] = out["list_status"].fillna("").astype(str)
        out["is_st"] = out["is_st"].fillna(0).astype(int)
        out = out[required_cols]
        out = out.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        return out

    def _fetch_stock_info_tushare(self) -> pd.DataFrame | None:
        self._require_pro_api("get_stock_info(tushare)")
        frames = []
        for status in ["L", "D", "P"]:
            part = self.pro.stock_basic(
                exchange='',
                list_status=status,
                fields='ts_code,name,list_date,delist_date,list_status',
            )
            time.sleep(API_DELAY)
            if part is not None and not part.empty:
                frames.append(part)

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        # 防御性去重：保留上市状态优先级 L > P > D 的记录
        status_priority = {"L": 0, "P": 1, "D": 2}
        df["_status_priority"] = df["list_status"].map(status_priority).fillna(9)
        df = (
            df.sort_values(["ts_code", "_status_priority"])
            .drop_duplicates(subset=["ts_code"], keep="first")
            .drop(columns=["_status_priority"])
        )
        df = df[~df["ts_code"].str.endswith(".BJ")]
        df["is_st"] = df["name"].str.contains("ST", na=False).astype(int)
        return self._normalize_stock_info(df)

    def _fetch_stock_info_akshare(self) -> pd.DataFrame | None:
        # akshare 仅提供存续股票基础信息，不包含退市/停牌全量信息
        raw = None
        try:
            raw = ak.stock_info_a_code_name()
        except Exception:
            raw = None

        if raw is None or raw.empty:
            try:
                raw = ak.stock_zh_a_spot_em()[["代码", "名称"]]
            except Exception:
                return None

        # 兼容不同字段名
        code_col = "code" if "code" in raw.columns else "代码"
        name_col = "name" if "name" in raw.columns else "名称"
        if code_col not in raw.columns or name_col not in raw.columns:
            return None

        df = raw[[code_col, name_col]].copy()
        df.columns = ["code", "name"]
        df["code"] = df["code"].astype(str).str.zfill(6)
        # 北交所代码通常以 4/8 开头，暂与当前逻辑保持一致直接排除
        df = df[~df["code"].str.startswith(("4", "8"))]

        def to_ts_code(code: str) -> str:
            if code.startswith(("6", "9", "5")):
                return f"{code}.SH"
            return f"{code}.SZ"

        df["ts_code"] = df["code"].map(to_ts_code)
        df["is_st"] = df["name"].str.contains("ST", na=False).astype(int)
        df["list_date"] = ""
        df["delist_date"] = ""
        df["list_status"] = "L"
        df = df.drop(columns=["code"])
        return self._normalize_stock_info(df)

    # ===== 股票基本信息 =====

    def get_stock_info(self) -> pd.DataFrame:
        """获取全市场股票基本信息（含退市/暂停），用于 ST 过滤和上市日期判断"""
        cache_path = os.path.join(DATA_DIR, "stock_info.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            required_cols = {"ts_code", "name", "list_date", "delist_date", "list_status", "is_st"}
            if required_cols.issubset(set(cached.columns)):
                print(f"[DataAgent] 股票信息已缓存，共 {len(cached)} 只")
                return cached

            if self.pro is None:
                # 无 token 时无法刷新远程缓存：补齐缺失列，保持兼容运行
                cached = self._normalize_stock_info(cached)
                print("[DataAgent] 警告: stock_info缓存字段不完整，且未设置 TUSHARE_TOKEN，使用兼容模式缓存。")
                return cached

            print("[DataAgent] 检测到旧版stock_info缓存，准备自动刷新...")

        order = ["tushare", "akshare"] if self.stock_info_provider == "auto" else [self.stock_info_provider]
        for provider in order:
            try:
                print(f"[DataAgent] 正在获取股票基本信息（provider={provider}）...")
                if provider == "tushare":
                    df = self._fetch_stock_info_tushare()
                else:
                    df = self._fetch_stock_info_akshare()

                if df is not None and not df.empty:
                    save_parquet(df, cache_path)
                    st_count = int(df["is_st"].sum()) if "is_st" in df.columns else 0
                    print(
                        f"[DataAgent] 股票信息获取完成(provider={provider}): "
                        f"{len(df)} 只, 其中 ST {st_count} 只"
                    )
                    return df
            except Exception as e:
                print(f"[DataAgent] 股票信息获取失败(provider={provider}): {e}")

        if cached is not None:
            cached = self._normalize_stock_info(cached)
            print("[DataAgent] 回退使用本地stock_info缓存（兼容模式）。")
            return cached

        # 兜底：无法获取时返回空 DataFrame
        print("[DataAgent] 警告: 无法获取股票信息，跳过 ST 过滤")
        return pd.DataFrame(
            columns=["ts_code", "name", "list_date", "delist_date", "list_status", "is_st"]
        )

    # ===== 交易日历 =====

    def _get_trade_dates_from_akshare(self) -> list[str]:
        """使用AKShare指数日线近似交易日历。"""
        df = ak.stock_zh_index_daily(symbol="sh000852")
        if df is None or df.empty:
            raise RuntimeError("[DataAgent] AKShare无法获取交易日历")
        dates = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        dates = dates[(dates >= START_DATE) & (dates <= END_DATE)]
        return sorted(dates.unique().tolist())

    def _get_trade_dates_from_daily(self) -> list[str]:
        """通过 daily 接口获取交易日列表"""
        cache_path = os.path.join(DATA_DIR, "trade_dates.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            return cached["trade_date"].tolist()

        dates = []
        if self.pro is not None:
            print("[DataAgent] 正在获取交易日历（provider=tushare）...")
            try:
                df = self.pro.daily(ts_code="000001.SZ", start_date=START_DATE, end_date=END_DATE)
                time.sleep(API_DELAY)
                if df is not None and not df.empty:
                    dates = sorted(df["trade_date"].unique().tolist())
            except Exception as e:
                print(f"[DataAgent] 交易日历获取失败(provider=tushare): {e}")

        if not dates:
            print("[DataAgent] 正在获取交易日历（provider=akshare）...")
            dates = self._get_trade_dates_from_akshare()

        df_dates = pd.DataFrame({"trade_date": dates})
        save_parquet(df_dates, cache_path)
        print(f"[DataAgent] 交易日历获取完成，共 {len(dates)} 个交易日")
        return dates

    # ===== 行情数据 =====

    def _fetch_raw_quotes(self) -> pd.DataFrame:
        """拉取原始行情（无清洗）"""
        cache_path = os.path.join(DATA_DIR, "daily_quotes_raw.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] 原始行情已缓存，共 {len(cached)} 行")
            return cached

        self._require_pro_api("_fetch_raw_quotes")
        trade_dates = self._get_trade_dates_from_daily()
        print(f"[DataAgent] 按日期拉取全市场行情，共 {len(trade_dates)} 个交易日")
        print(f"  预计耗时: {len(trade_dates) * API_DELAY / 60:.1f} 分钟")

        all_data = []
        failed_dates = []
        for i, date in enumerate(trade_dates):
            try:
                df = self.pro.daily(trade_date=date)
                time.sleep(API_DELAY)

                if df is not None and not df.empty:
                    df = df[~df["ts_code"].str.endswith(".BJ")]
                    all_data.append(df[["ts_code", "trade_date",
                                        "open", "high", "low", "close",
                                        "vol", "amount", "pre_close"]])

                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{len(trade_dates)} 个交易日")

            except Exception as e:
                print(f"  {date} 拉取失败: {e}")
                failed_dates.append(date)
                time.sleep(2)
                continue

        # 重试失败的日期（最多3轮）
        for retry in range(3):
            if not failed_dates:
                break
            print(f"  重试第 {retry + 1} 轮，共 {len(failed_dates)} 个失败日期")
            still_failed = []
            for date in failed_dates:
                try:
                    time.sleep(2)
                    df = self.pro.daily(trade_date=date)
                    time.sleep(API_DELAY)
                    if df is not None and not df.empty:
                        df = df[~df["ts_code"].str.endswith(".BJ")]
                        all_data.append(df[["ts_code", "trade_date",
                                            "open", "high", "low", "close",
                                            "vol", "amount", "pre_close"]])
                        print(f"    {date} 重试成功")
                    else:
                        still_failed.append(date)
                except Exception:
                    still_failed.append(date)
            failed_dates = still_failed

        if failed_dates:
            print(f"  警告: {len(failed_dates)} 个日期最终失败: {failed_dates}")

        print("[DataAgent] 合并行情数据...")
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        for col in ["open", "high", "low", "close", "vol", "amount", "pre_close"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        save_parquet(result, cache_path)
        print(f"[DataAgent] 原始行情拉取完成，共 {len(result)} 行")
        return result

    def _get_adj_factors(self) -> pd.DataFrame:
        """获取复权因子（按配置日期范围缓存）。"""
        cache_path = os.path.join(DATA_DIR, "adj_factors.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] 复权因子已缓存，共 {len(cached)} 行")
            return cached

        self._require_pro_api("_get_adj_factors")
        trade_dates = self._get_trade_dates_from_daily()
        print(f"[DataAgent] 按日期拉取复权因子，共 {len(trade_dates)} 个交易日")

        all_data = []
        failed_dates = []
        for i, date in enumerate(trade_dates):
            try:
                df = self.pro.adj_factor(trade_date=date, fields="ts_code,trade_date,adj_factor")
                time.sleep(API_DELAY)

                if df is not None and not df.empty:
                    df = df[~df["ts_code"].str.endswith(".BJ")]
                    all_data.append(df)

                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{len(trade_dates)} 个交易日")

            except Exception as e:
                print(f"  {date} 拉取失败: {e}")
                failed_dates.append(date)
                time.sleep(2)
                continue

        for retry in range(2):
            if not failed_dates:
                break
            print(f"  复权因子重试第 {retry + 1} 轮，共 {len(failed_dates)} 个失败日期")
            still_failed = []
            for date in failed_dates:
                try:
                    time.sleep(2)
                    df = self.pro.adj_factor(trade_date=date, fields="ts_code,trade_date,adj_factor")
                    time.sleep(API_DELAY)
                    if df is not None and not df.empty:
                        df = df[~df["ts_code"].str.endswith(".BJ")]
                        all_data.append(df)
                        print(f"    {date} 重试成功")
                    else:
                        still_failed.append(date)
                except Exception:
                    still_failed.append(date)
            failed_dates = still_failed

        if not all_data:
            raise RuntimeError("[DataAgent] 无法获取任何复权因子数据")

        if failed_dates:
            print(f"  警告: {len(failed_dates)} 个日期复权因子最终失败: {failed_dates}")

        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        result["adj_factor"] = pd.to_numeric(result["adj_factor"], errors="coerce")

        save_parquet(result, cache_path)
        print(f"[DataAgent] 复权因子拉取完成，共 {len(result)} 行")
        return result

    def _apply_price_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """对价格字段应用前/后复权。"""
        if self.price_adj_mode == "none":
            print("[DataAgent] PRICE_ADJ_MODE=none，跳过价格复权")
            return df

        adj_df = self._get_adj_factors()
        work = df.merge(adj_df, on=["ts_code", "trade_date"], how="left")
        work["adj_factor"] = pd.to_numeric(work["adj_factor"], errors="coerce")
        work["adj_factor"] = work.groupby("ts_code")["adj_factor"].transform(lambda x: x.ffill().bfill())

        if self.price_adj_mode == "qfq":
            anchor = work.groupby("ts_code")["adj_factor"].transform("last")
        else:  # hfq
            anchor = work.groupby("ts_code")["adj_factor"].transform("first")

        scale = work["adj_factor"] / anchor
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        price_cols = ["open", "high", "low", "close", "pre_close"]
        for col in price_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce") * scale

        work = work.drop(columns=["adj_factor"])
        print(f"[DataAgent] 已应用价格复权: {self.price_adj_mode}")
        return work

    def get_daily_quotes(self) -> pd.DataFrame:
        """
        获取清洗后的日线行情

        清洗步骤：
        1. 过滤 ST 股票
        2. 标记 IPO 首日（上市后前5个交易日）
        3. 标记涨跌停
        4. 应用价格复权（qfq/hfq/none）
        """
        cache_path = os.path.join(DATA_DIR, "daily_quotes.parquet")
        meta_path = os.path.join(DATA_DIR, "daily_quotes.meta.json")
        cached = load_parquet(cache_path)
        if cached is not None:
            meta = self._load_cache_meta(meta_path)
            cached_mode = str(meta.get("price_adj_mode", "")).lower()
            if cached_mode == self.price_adj_mode:
                print(f"[DataAgent] 日线行情已缓存，共 {len(cached)} 行（PRICE_ADJ_MODE={cached_mode}）")
                return cached

            if self.pro is None:
                print(
                    "[DataAgent] 警告: 日线缓存复权模式不匹配，且未设置 TUSHARE_TOKEN。"
                    "继续使用现有缓存。"
                )
                print(f"[DataAgent] 日线行情已缓存，共 {len(cached)} 行（legacy）")
                return cached

            print(
                f"[DataAgent] 日线缓存复权模式不匹配（cached={cached_mode or 'unknown'}, "
                f"current={self.price_adj_mode}），将重新构建。"
            )

        # 拉取原始数据
        raw = self._fetch_raw_quotes()
        stock_info = self.get_stock_info()

        print("[DataAgent] 开始数据清洗...")
        df = raw.copy()
        n_before = len(df)

        # 1. 过滤 ST 股票
        if len(stock_info) > 0 and "is_st" in stock_info.columns:
            st_codes = stock_info[stock_info["is_st"] == 1]["ts_code"].tolist()
            df = df[~df["ts_code"].isin(st_codes)]
            n_after_st = len(df)
            print(f"  过滤 ST: {n_before - n_after_st} 行 ({len(st_codes)} 只 ST 股)")
        else:
            n_after_st = n_before

        # 2. 标记 IPO 期（上市后前5个交易日，不参与因子计算）
        if len(stock_info) > 0 and "list_date" in stock_info.columns:
            df = df.merge(
                stock_info[["ts_code", "list_date"]],
                on="ts_code", how="left"
            )
            df["list_date"] = df["list_date"].fillna("").astype(str)

            # 仅对样本期内新上市股票标记 IPO 前5个交易日，避免样本起点误标老股票
            has_list_date = df["list_date"].str.len().eq(8)
            listed_in_sample = has_list_date & (df["list_date"] >= START_DATE)
            after_list = has_list_date & (df["trade_date"] >= df["list_date"])

            df["_ipo_rank"] = np.nan
            df.loc[after_list, "_ipo_rank"] = (
                df.loc[after_list].groupby("ts_code").cumcount() + 1
            )
            df["is_ipo_period"] = (
                listed_in_sample & after_list & (df["_ipo_rank"] <= 5)
            ).astype(int)

            df = df.drop(columns=["list_date", "_ipo_rank"])
            ipo_rows = df["is_ipo_period"].sum()
            print(f"  标记 IPO 期: {ipo_rows} 行（样本内新股上市后前5个交易日）")
        else:
            df["is_ipo_period"] = 0

        # 3. 标记涨跌停（用 pre_close 判断，区分不同板块涨跌停幅度）
        if "pre_close" in df.columns and df["pre_close"].notna().sum() > 0:
            df["pct_chg"] = (df["close"] / df["pre_close"] - 1) * 100

            # 确定每只股票的涨跌停阈值
            # 科创板(688开头): ±20%  创业板(300开头,2020-08-24起): ±20%  其他: ±10%
            is_star = df["ts_code"].str.startswith("688")  # 科创板
            is_gem = df["ts_code"].str.startswith("300")   # 创业板
            gem_reform = df["trade_date"] >= "20200824"     # 创业板注册制日期

            limit_pct = pd.Series(9.5, index=df.index)     # 默认主板 10%（留精度余量）
            limit_pct[is_star] = 19.5                       # 科创板 20%
            limit_pct[is_gem & gem_reform] = 19.5           # 创业板 20%（注册制后）

            df["is_limit_up"] = (df["pct_chg"] >= limit_pct).astype(int)
            df["is_limit_down"] = (df["pct_chg"] <= -limit_pct).astype(int)
            limit_up = df["is_limit_up"].sum()
            limit_down = df["is_limit_down"].sum()
            print(f"  标记涨停: {limit_up} 行, 跌停: {limit_down} 行（含科创板/创业板20%）")
        else:
            df["pct_chg"] = 0
            df["is_limit_up"] = 0
            df["is_limit_down"] = 0

        # 4. 价格复权（在涨跌停标记之后，避免影响真实交易规则识别）
        df = self._apply_price_adjustment(df)

        n_final = len(df)
        print(f"  清洗完成: {n_before} → {n_final} 行 (移除 {n_before - n_final} 行)")

        save_parquet(df, cache_path)
        self._save_cache_meta(
            meta_path,
            {
                "price_adj_mode": self.price_adj_mode,
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
        )
        return df

    # ===== 基准指数 =====

    def get_benchmark(self) -> pd.DataFrame:
        """获取中证1000指数日线（AKShare）"""
        cache_path = os.path.join(DATA_DIR, "benchmark.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] 基准指数已缓存，共 {len(cached)} 行")
            return cached

        print("[DataAgent] 正在拉取中证1000指数行情（AKShare）...")
        df = ak.stock_zh_index_daily(symbol="sh000852")

        if df is not None and not df.empty:
            df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
            df = df.rename(columns={"close": "bench_close"})
            df = df[["trade_date", "bench_close"]]
            df = df[(df["trade_date"] >= START_DATE) & (df["trade_date"] <= END_DATE)]
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["bench_close"] = pd.to_numeric(df["bench_close"], errors="coerce")
            save_parquet(df, cache_path)
            print(f"[DataAgent] 基准指数拉取完成，共 {len(df)} 行")
            return df
        else:
            raise RuntimeError("[DataAgent] 无法获取基准指数数据")

    # ===== 主流程 =====

    def update_all(self) -> dict:
        """一键拉取并清洗所有数据"""
        print("=" * 50)
        print("[DataAgent] 开始全量数据拉取")
        print("=" * 50)

        quotes = self.get_daily_quotes()
        benchmark = self.get_benchmark()

        summary = {
            "股票数量": quotes["ts_code"].nunique(),
            "行情数据行数": len(quotes),
            "基准数据行数": len(benchmark),
            "日期范围": f"{START_DATE} ~ {END_DATE}",
        }

        # 数据质量报告
        stock_info = load_parquet(os.path.join(DATA_DIR, "stock_info.parquet"))
        if stock_info is not None and "is_st" in stock_info.columns:
            summary["ST过滤"] = f"已完成（ST样本 {int(stock_info['is_st'].sum())} 只）"
        else:
            summary["ST过滤"] = "未提供（缺少stock_info缓存）"
        if "is_ipo_period" in quotes.columns:
            summary["IPO期标记"] = f"{quotes['is_ipo_period'].sum()} 行"
        if "is_limit_up" in quotes.columns:
            summary["涨停标记"] = f"{quotes['is_limit_up'].sum()} 行"

        print("\n[DataAgent] 数据拉取完成:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        self.emit("data.ready", summary)
        return summary


if __name__ == "__main__":
    agent = DataAgent()
    agent.update_all()
