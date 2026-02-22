"""
市场状态识别模块

通过动量 + 波动率指标识别市场状态（牛市/熊市/震荡）。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger
from utils.helpers import ensure_dir

logger = get_logger("analytics.regime")


class RegimeDetector:
    """市场状态检测器"""

    def detect(
        self,
        benchmark_df: pd.DataFrame,
        date: str,
        lookback_short: int = 20,
        lookback_long: int = 60,
    ) -> str:
        """
        检测指定日期的市场状态

        规则:
        - Bull: 短期动量 > 0 且 长期动量 > 0 且 波动率 < 历史中位数
        - Bear: 短期动量 < 0 且 长期动量 < 0
        - Sideways: 其他

        参数:
            benchmark_df: 含 trade_date, bench_close
            date: 目标日期

        返回: "bull" | "bear" | "sideways"
        """
        df = benchmark_df[benchmark_df["trade_date"] <= date].copy()
        df = df.sort_values("trade_date").reset_index(drop=True)

        if len(df) < lookback_long + 10:
            return "sideways"

        close = df["bench_close"].values

        # 短期动量
        short_mom = close[-1] / close[-lookback_short] - 1 if len(close) > lookback_short else 0
        # 长期动量
        long_mom = close[-1] / close[-lookback_long] - 1 if len(close) > lookback_long else 0

        # 波动率（20 日日收益率标准差）
        returns = pd.Series(close).pct_change().dropna()
        recent_vol = returns.iloc[-lookback_short:].std() if len(returns) > lookback_short else 0
        median_vol = returns.rolling(lookback_long).std().median() if len(returns) > lookback_long else recent_vol

        if short_mom > 0 and long_mom > 0 and recent_vol < median_vol * 1.2:
            return "bull"
        elif short_mom < 0 and long_mom < 0:
            return "bear"
        else:
            return "sideways"

    def get_history(self, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算全历史的状态序列

        返回: DataFrame[trade_date, regime, regime_changed]
        """
        df = benchmark_df.sort_values("trade_date").reset_index(drop=True)
        dates = df["trade_date"].tolist()

        records = []
        prev_regime = None

        for date in dates:
            regime = self.detect(benchmark_df, date)
            changed = regime != prev_regime
            records.append({
                "trade_date": date,
                "regime": regime,
                "regime_changed": changed,
            })
            prev_regime = regime

        return pd.DataFrame(records)

    @staticmethod
    def plot_regime_overlay(
        benchmark_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        output_dir: str,
    ):
        """状态叠加图"""
        ensure_dir(output_dir)

        fig, ax = plt.subplots(figsize=(14, 6))
        dates = pd.to_datetime(benchmark_df["trade_date"], format="%Y%m%d")
        ax.plot(dates, benchmark_df["bench_close"], color="black", linewidth=1, label="Benchmark")

        # 状态背景着色
        regime_colors = {"bull": "#2ecc71", "bear": "#e74c3c", "sideways": "#f39c12"}
        merged = benchmark_df.merge(regime_df[["trade_date", "regime"]], on="trade_date", how="left")
        merged["regime"] = merged["regime"].fillna("sideways")
        merged_dates = pd.to_datetime(merged["trade_date"], format="%Y%m%d")

        for regime, color in regime_colors.items():
            mask = merged["regime"] == regime
            if mask.any():
                ax.fill_between(
                    merged_dates, merged["bench_close"].min(), merged["bench_close"].max(),
                    where=mask, alpha=0.15, color=color, label=regime.capitalize(),
                )

        ax.set_title("Market Regime Detection")
        ax.set_xlabel("Date")
        ax.set_ylabel("Benchmark Price")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "regime_detection.png"), dpi=150)
        plt.close(fig)
        logger.info("状态检测图已保存: regime_detection.png")
