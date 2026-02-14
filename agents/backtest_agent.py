"""
Backtest Agent - 历史回测
负责模拟月度调仓策略的历史表现
"""
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_DIR, OUTPUT_DIR, TRADE_COST_BUY, TRADE_COST_SELL
from utils.helpers import ensure_dir
from core.agent_base import BaseAgent


class BacktestAgent(BaseAgent):
    """回测Agent：月度调仓回测"""

    agent_name = "BacktestAgent"

    def __init__(self, daily_quotes: pd.DataFrame, benchmark: pd.DataFrame):
        super().__init__()
        self.quotes = daily_quotes[["ts_code", "trade_date", "close"]].copy()
        self.benchmark = benchmark.copy()
        # 保留跌停标记用于卖出限制
        if "is_limit_down" in daily_quotes.columns:
            self.limit_down = daily_quotes[["ts_code", "trade_date", "is_limit_down"]].copy()
        else:
            self.limit_down = None
        print("[BacktestAgent] 初始化完成")

    def run(self, holdings_by_date: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        执行回测（防前视偏差版）
        参数：
            holdings_by_date: {调仓日期: DataFrame(ts_code, weight)}
        返回：
            每日净值 DataFrame

        关键：调仓日 T 的信号基于 T 日收盘价计算，新持仓从 T+1 日开始生效。
        T 日的收益仍然使用旧持仓计算，避免前视偏差。
        """
        print(f"[BacktestAgent] 开始回测，共 {len(holdings_by_date)} 个调仓期...")

        all_trade_dates = sorted(self.quotes["trade_date"].unique())

        nav_records = []
        current_nav = 1.0
        current_holdings = None
        pending_holdings = None  # 待生效的新持仓
        total_cost = 0.0

        for i, td in enumerate(all_trade_dates):
            # 先应用上一日挂起的调仓（T+1 生效）
            if pending_holdings is not None:
                # 计算换手成本
                cost = self._calc_turnover_cost(current_holdings, pending_holdings)
                current_nav *= (1 - cost)
                total_cost += cost
                current_holdings = pending_holdings
                pending_holdings = None

            # 如果今天是调仓日，挂起新持仓（明天生效）
            if td in holdings_by_date:
                pending_holdings = holdings_by_date[td]

            if current_holdings is None:
                continue

            # 计算持仓组合的日收益率
            today_prices = self.quotes[self.quotes["trade_date"] == td][["ts_code", "close"]]

            if i > 0:
                prev_td = all_trade_dates[i - 1]
                prev_prices = self.quotes[self.quotes["trade_date"] == prev_td][["ts_code", "close"]]
                prev_prices = prev_prices.rename(columns={"close": "prev_close"})

                merged = current_holdings.merge(today_prices, on="ts_code", how="inner")
                merged = merged.merge(prev_prices, on="ts_code", how="inner")

                if len(merged) > 0:
                    merged["stock_ret"] = merged["close"] / merged["prev_close"] - 1
                    portfolio_ret = (merged["weight"] * merged["stock_ret"]).sum()
                    current_nav *= (1 + portfolio_ret)

            nav_records.append({"trade_date": td, "nav": current_nav})

        print(f"[BacktestAgent] 累计交易成本: {total_cost:.2%}")

        nav_df = pd.DataFrame(nav_records)

        # 合并基准
        nav_df = nav_df.merge(self.benchmark, on="trade_date", how="left")
        nav_df["bench_close"] = nav_df["bench_close"].ffill()
        if nav_df["bench_close"].iloc[0] > 0:
            nav_df["bench_nav"] = nav_df["bench_close"] / nav_df["bench_close"].iloc[0]
        else:
            nav_df["bench_nav"] = 1.0

        # 超额净值
        nav_df["excess_nav"] = nav_df["nav"] / nav_df["bench_nav"]

        ensure_dir(OUTPUT_DIR)
        nav_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_nav.csv"),
                       index=False, encoding="utf-8-sig")

        print("[BacktestAgent] 回测完成")
        return nav_df

    @staticmethod
    def _calc_turnover_cost(old_holdings, new_holdings) -> float:
        """
        计算调仓换手成本
        买入成本 TRADE_COST_BUY，卖出成本 TRADE_COST_SELL
        """
        if old_holdings is None:
            # 首次建仓，只有买入成本
            return TRADE_COST_BUY

        old = old_holdings.set_index("ts_code")["weight"]
        new = new_holdings.set_index("ts_code")["weight"]

        # 合并新旧持仓
        all_codes = old.index.union(new.index)
        old_w = old.reindex(all_codes, fill_value=0)
        new_w = new.reindex(all_codes, fill_value=0)

        # 买入量和卖出量
        delta = new_w - old_w
        buy_amount = delta[delta > 0].sum()    # 新买入的权重
        sell_amount = (-delta[delta < 0]).sum()  # 卖出的权重

        cost = buy_amount * TRADE_COST_BUY + sell_amount * TRADE_COST_SELL
        return cost

    @staticmethod
    def calc_metrics(nav_df: pd.DataFrame) -> dict:
        """计算回测绩效指标"""
        nav = nav_df["nav"].values
        bench_nav = nav_df["bench_nav"].values

        # 年化收益
        n_days = len(nav)
        total_ret = nav[-1] / nav[0] - 1
        ann_ret = (1 + total_ret) ** (252 / n_days) - 1

        # 基准年化收益
        bench_total = bench_nav[-1] / bench_nav[0] - 1
        bench_ann = (1 + bench_total) ** (252 / n_days) - 1

        # 超额年化
        excess_ann = ann_ret - bench_ann

        # 日收益率
        daily_ret = pd.Series(nav).pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252)

        # 夏普比率（无风险利率 2%）
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0

        # 最大回撤
        peak = pd.Series(nav).cummax()
        drawdown = (pd.Series(nav) - peak) / peak
        max_dd = drawdown.min()

        # 超额最大回撤
        excess = nav_df["excess_nav"].values
        excess_peak = pd.Series(excess).cummax()
        excess_dd = (pd.Series(excess) - excess_peak) / excess_peak
        excess_max_dd = excess_dd.min()

        metrics = {
            "总收益率": f"{total_ret:.2%}",
            "年化收益率": f"{ann_ret:.2%}",
            "基准年化收益率": f"{bench_ann:.2%}",
            "年化超额收益": f"{excess_ann:.2%}",
            "年化波动率": f"{ann_vol:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "超额最大回撤": f"{excess_max_dd:.2%}",
        }
        return metrics
