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
import time
import tushare as ts
import akshare as ak
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TUSHARE_TOKEN, DATA_DIR, START_DATE, END_DATE, API_DELAY
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
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        ensure_dir(DATA_DIR)
        print("[DataAgent] 初始化完成")

    # ===== 股票基本信息 =====

    def get_stock_info(self) -> pd.DataFrame:
        """获取全市场股票基本信息（名称、上市日期），用于 ST 过滤"""
        cache_path = os.path.join(DATA_DIR, "stock_info.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] 股票信息已缓存，共 {len(cached)} 只")
            return cached

        print("[DataAgent] 正在获取股票基本信息...")
        try:
            df = self.pro.stock_basic(
                exchange='', list_status='L',
                fields='ts_code,name,list_date'
            )
            time.sleep(API_DELAY)

            if df is not None and not df.empty:
                # 排除北交所
                df = df[~df["ts_code"].str.endswith(".BJ")]
                # 标记 ST
                df["is_st"] = df["name"].str.contains("ST", na=False).astype(int)
                save_parquet(df, cache_path)
                st_count = df["is_st"].sum()
                print(f"[DataAgent] 股票信息获取完成: {len(df)} 只, 其中 ST {st_count} 只")
                return df
        except Exception as e:
            print(f"[DataAgent] 股票信息获取失败: {e}")

        # 兜底：无法获取时返回空 DataFrame
        print("[DataAgent] 警告: 无法获取股票信息，跳过 ST 过滤")
        return pd.DataFrame(columns=["ts_code", "name", "list_date", "is_st"])

    # ===== 交易日历 =====

    def _get_trade_dates_from_daily(self) -> list[str]:
        """通过 daily 接口获取交易日列表"""
        cache_path = os.path.join(DATA_DIR, "trade_dates.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            return cached["trade_date"].tolist()

        print("[DataAgent] 正在获取交易日历...")
        df = self.pro.daily(ts_code="000001.SZ", start_date=START_DATE, end_date=END_DATE)
        time.sleep(API_DELAY)

        if df is not None and not df.empty:
            dates = sorted(df["trade_date"].unique().tolist())
            df_dates = pd.DataFrame({"trade_date": dates})
            save_parquet(df_dates, cache_path)
            print(f"[DataAgent] 交易日历获取完成，共 {len(dates)} 个交易日")
            return dates
        else:
            raise RuntimeError("[DataAgent] 无法获取交易日历")

    # ===== 行情数据 =====

    def _fetch_raw_quotes(self) -> pd.DataFrame:
        """拉取原始行情（无清洗）"""
        cache_path = os.path.join(DATA_DIR, "daily_quotes_raw.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] 原始行情已缓存，共 {len(cached)} 行")
            return cached

        trade_dates = self._get_trade_dates_from_daily()
        print(f"[DataAgent] 按日期拉取全市场行情，共 {len(trade_dates)} 个交易日")
        print(f"  预计耗时: {len(trade_dates) * API_DELAY / 60:.1f} 分钟")

        all_data = []
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
                time.sleep(2)
                continue

        print("[DataAgent] 合并行情数据...")
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        for col in ["open", "high", "low", "close", "vol", "amount", "pre_close"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        save_parquet(result, cache_path)
        print(f"[DataAgent] 原始行情拉取完成，共 {len(result)} 行")
        return result

    def get_daily_quotes(self) -> pd.DataFrame:
        """
        获取清洗后的日线行情

        清洗步骤：
        1. 过滤 ST 股票
        2. 标记 IPO 首日（上市后前5个交易日）
        3. 标记涨跌停
        4. 计算日收益率
        """
        cache_path = os.path.join(DATA_DIR, "daily_quotes.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] 日线行情已缓存，共 {len(cached)} 行")
            return cached

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

        # 2. 标记 IPO 首日（上市后前5个交易日，不参与因子计算）
        if len(stock_info) > 0 and "list_date" in stock_info.columns:
            df = df.merge(
                stock_info[["ts_code", "list_date"]],
                on="ts_code", how="left"
            )
            # 计算上市后第几个交易日
            df["_rank"] = df.groupby("ts_code")["trade_date"].rank(method="first")
            df["is_ipo_period"] = (df["_rank"] <= 5).astype(int)
            df = df.drop(columns=["list_date", "_rank"])
            ipo_rows = df["is_ipo_period"].sum()
            print(f"  标记 IPO 期: {ipo_rows} 行（上市前5日）")
        else:
            df["is_ipo_period"] = 0

        # 3. 标记涨跌停（用 pre_close 判断）
        if "pre_close" in df.columns and df["pre_close"].notna().sum() > 0:
            df["pct_chg"] = (df["close"] / df["pre_close"] - 1) * 100
            # 涨停：涨幅 >= 9.5%（考虑精度）  跌停：跌幅 <= -9.5%
            df["is_limit_up"] = (df["pct_chg"] >= 9.5).astype(int)
            df["is_limit_down"] = (df["pct_chg"] <= -9.5).astype(int)
            limit_up = df["is_limit_up"].sum()
            limit_down = df["is_limit_down"].sum()
            print(f"  标记涨停: {limit_up} 行, 跌停: {limit_down} 行")
        else:
            df["pct_chg"] = 0
            df["is_limit_up"] = 0
            df["is_limit_down"] = 0

        n_final = len(df)
        print(f"  清洗完成: {n_before} → {n_final} 行 (移除 {n_before - n_final} 行)")

        save_parquet(df, cache_path)
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
        if "is_st" not in quotes.columns:
            summary["ST过滤"] = "已完成"
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
