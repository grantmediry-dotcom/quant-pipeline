"""
绩效归因模块

提供因子贡献分解、交易成本拆分、月度 PnL 瀑布图。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger
from utils.helpers import ensure_dir

logger = get_logger("analytics.attribution")


class PerformanceAttributor:
    """绩效归因分析器"""

    def factor_pnl_decomposition(
        self,
        holdings_history: list[dict],
        factor_scores_history: dict[str, pd.DataFrame],
        factor_weights: dict[str, float],
        daily_quotes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        因子贡献分解：逐期计算各因子对组合收益的贡献

        方法：
        每期组合的因子暴露 = Σ(w_i × factor_score_i)
        因子贡献 ≈ 因子暴露 × 因子 IC 方向 × 实际收益

        参数:
            holdings_history: [{date, holdings: DataFrame(ts_code, weight)}, ...]
            factor_scores_history: {date: DataFrame(ts_code, factor_1, factor_2, ...)}
            factor_weights: {factor_name: weight_in_alpha}
            daily_quotes: 日线行情

        返回:
            DataFrame[date, total_ret, factor_contrib_{name}, cost_drag, unexplained]
        """
        if not holdings_history:
            return pd.DataFrame()

        factor_names = list(factor_weights.keys())
        records = []

        for entry in holdings_history:
            date = entry["date"]
            holdings = entry["holdings"]  # DataFrame(ts_code, weight)
            cost = entry.get("cost", 0.0)

            # 组合实际收益（从 NAV 变化中取，此处用持仓价格近似）
            total_ret = self._calc_period_return(holdings, daily_quotes, date)

            # 因子暴露与贡献
            factor_contribs = {}
            scores_df = factor_scores_history.get(date)

            if scores_df is not None and not scores_df.empty:
                merged = holdings.merge(scores_df, on="ts_code", how="left")
                total_factor_contrib = 0.0

                for fname in factor_names:
                    if fname in merged.columns:
                        # 加权因子暴露
                        exposure = (merged["weight"] * merged[fname].fillna(0)).sum()
                        w = factor_weights.get(fname, 0.0)
                        contrib = exposure * w * total_ret if total_ret != 0 else 0.0
                        factor_contribs[f"factor_{fname}"] = round(contrib, 6)
                        total_factor_contrib += contrib
                    else:
                        factor_contribs[f"factor_{fname}"] = 0.0

                unexplained = total_ret - total_factor_contrib + cost
            else:
                for fname in factor_names:
                    factor_contribs[f"factor_{fname}"] = 0.0
                unexplained = total_ret + cost

            record = {
                "date": date,
                "total_ret": round(total_ret, 6),
                "cost_drag": round(-cost, 6),
                "unexplained": round(unexplained, 6),
                **factor_contribs,
            }
            records.append(record)

        return pd.DataFrame(records)

    def cost_breakdown(self, cost_history: list[dict]) -> pd.DataFrame:
        """
        交易成本拆分

        参数:
            cost_history: [{date, buy_cost, sell_cost, total_cost}, ...]

        返回:
            DataFrame[date, buy_cost, sell_cost, total_cost]
        """
        if not cost_history:
            return pd.DataFrame()
        return pd.DataFrame(cost_history)

    def period_waterfall(self, nav_df: pd.DataFrame) -> pd.DataFrame:
        """
        月度 PnL 瀑布图数据

        参数:
            nav_df: 含 trade_date, nav 的 DataFrame

        返回:
            DataFrame[month, pnl, cumulative]
        """
        df = nav_df.copy()
        df["month"] = df["trade_date"].str[:6]

        monthly = df.groupby("month").agg(
            nav_start=("nav", "first"),
            nav_end=("nav", "last"),
        ).reset_index()

        monthly["pnl"] = monthly["nav_end"] / monthly["nav_start"] - 1
        monthly["cumulative"] = (1 + monthly["pnl"]).cumprod() - 1

        return monthly

    def save_report(self, attribution_df: pd.DataFrame, nav_df: pd.DataFrame, output_dir: str):
        """保存归因报告"""
        ensure_dir(output_dir)

        # CSV
        if not attribution_df.empty:
            attribution_df.to_csv(
                os.path.join(output_dir, "attribution_report.csv"),
                index=False, encoding="utf-8-sig",
            )
            logger.info("归因报告已保存: attribution_report.csv")

        # 月度瀑布图
        waterfall = self.period_waterfall(nav_df)
        if not waterfall.empty:
            waterfall.to_csv(
                os.path.join(output_dir, "monthly_pnl.csv"),
                index=False, encoding="utf-8-sig",
            )
            self._plot_waterfall(waterfall, output_dir)

    def _plot_waterfall(self, waterfall: pd.DataFrame, output_dir: str):
        """绘制月度 PnL 瀑布图"""
        fig, ax = plt.subplots(figsize=(14, 6))

        months = waterfall["month"].values
        pnl = waterfall["pnl"].values
        colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in pnl]

        ax.bar(range(len(months)), pnl * 100, color=colors, width=0.7)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Monthly Return (%)")
        ax.set_title("Monthly PnL Waterfall")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "monthly_pnl_waterfall.png"), dpi=150)
        plt.close(fig)
        logger.info("月度 PnL 瀑布图已保存: monthly_pnl_waterfall.png")

    @staticmethod
    def _calc_period_return(
        holdings: pd.DataFrame,
        daily_quotes: pd.DataFrame,
        date: str,
        forward_days: int = 20,
    ) -> float:
        """计算持仓在未来 forward_days 天的加权收益"""
        all_dates = sorted(daily_quotes["trade_date"].unique())
        try:
            idx = list(all_dates).index(date)
        except ValueError:
            return 0.0

        end_idx = min(idx + forward_days, len(all_dates) - 1)
        end_date = all_dates[end_idx]

        start_prices = daily_quotes[daily_quotes["trade_date"] == date][["ts_code", "close"]]
        end_prices = daily_quotes[daily_quotes["trade_date"] == end_date][["ts_code", "close"]]
        end_prices = end_prices.rename(columns={"close": "close_end"})

        merged = holdings.merge(start_prices, on="ts_code", how="left")
        merged = merged.merge(end_prices, on="ts_code", how="left")

        valid = merged["close"].notna() & merged["close_end"].notna() & (merged["close"] > 0)
        if valid.sum() == 0:
            return 0.0

        merged.loc[valid, "ret"] = merged.loc[valid, "close_end"] / merged.loc[valid, "close"] - 1
        merged["ret"] = merged["ret"].fillna(0)

        return (merged["weight"] * merged["ret"]).sum()
