"""
Data Agent - 数据采集、清洗、本地缓存
混合数据源：Tushare daily(按日期) + AKShare index
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


class DataAgent:
    """
    数据Agent（混合数据源）
    - Tushare daily(trade_date): 按日期拉全市场行情（免费可用）
    - AKShare: 指数日线
    """

    def __init__(self):
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        ensure_dir(DATA_DIR)
        print("[DataAgent] 初始化完成")

    def _get_trade_dates_from_daily(self) -> list[str]:
        """通过 daily 接口获取交易日列表（从已拉取数据或采样获取）"""
        cache_path = os.path.join(DATA_DIR, "trade_dates.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            return cached["trade_date"].tolist()

        print("[DataAgent] 正在获取交易日历...")
        # 用一只活跃股票的交易记录来推算交易日
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

    def get_daily_quotes(self) -> pd.DataFrame:
        """
        按日期拉取全市场日线行情
        Tushare daily(trade_date=) 每次返回全市场数据，高效
        """
        cache_path = os.path.join(DATA_DIR, "daily_quotes.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] 日线行情已缓存，共 {len(cached)} 行")
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
                    # 过滤：排除北交所、ST（名称中含ST的后续再过滤）
                    df = df[~df["ts_code"].str.endswith(".BJ")]
                    all_data.append(df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]])

                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{len(trade_dates)} 个交易日")

            except Exception as e:
                print(f"  {date} 拉取失败: {e}")
                time.sleep(2)
                continue

        print("[DataAgent] 合并行情数据...")
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        # 确保数值类型
        for col in ["open", "high", "low", "close", "vol", "amount"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        save_parquet(result, cache_path)
        print(f"[DataAgent] 行情数据拉取完成，共 {len(result)} 行，涵盖 {result['ts_code'].nunique()} 只股票")
        return result

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

    def update_all(self) -> dict:
        """一键拉取所有数据"""
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
        print("\n[DataAgent] 数据拉取完成:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        return summary


if __name__ == "__main__":
    agent = DataAgent()
    agent.update_all()
