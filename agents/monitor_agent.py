"""
Monitor Agent - 绩效监控与报告生成
负责输出回测绩效报告和可视化图表
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无GUI后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUT_DIR
from utils.helpers import ensure_dir


class MonitorAgent:
    """监控Agent：绩效报告 + 可视化"""

    def __init__(self):
        ensure_dir(OUTPUT_DIR)
        print("[MonitorAgent] 初始化完成")

    def generate_report(self, metrics: dict, factor_test: pd.DataFrame, nav_df: pd.DataFrame):
        """生成完整绩效报告"""
        print("[MonitorAgent] 正在生成报告...")

        # 1. 绩效指标文本报告
        self._write_text_report(metrics, factor_test)

        # 2. 净值曲线图
        self._plot_nav_curve(nav_df)

        # 3. 超额净值曲线
        self._plot_excess_curve(nav_df)

        print(f"[MonitorAgent] 报告已生成至 {OUTPUT_DIR}")

    def _write_text_report(self, metrics: dict, factor_test: pd.DataFrame):
        """输出文本报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("    量化多因子策略 — 回测绩效报告")
        lines.append("=" * 60)
        lines.append("")
        lines.append("【一、策略绩效】")
        for k, v in metrics.items():
            lines.append(f"  {k:　<10}: {v}")

        lines.append("")
        lines.append("【二、单因子检验】")
        lines.append(factor_test.to_string(index=False))

        lines.append("")
        lines.append("【三、策略说明】")
        lines.append("  选股域: 中证1000成分股")
        lines.append("  因子: 动量20d、反转5d、换手率20d、波动率20d、市值")
        lines.append("  合成方式: 等权合成")
        lines.append("  选股: Top50 等权配置")
        lines.append("  调仓频率: 月度")
        lines.append("")
        lines.append("=" * 60)

        report_text = "\n".join(lines)
        print("\n" + report_text)

        with open(os.path.join(OUTPUT_DIR, "backtest_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_text)

    def _plot_nav_curve(self, nav_df: pd.DataFrame):
        """绘制策略 vs 基准净值曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))

        dates = pd.to_datetime(nav_df["trade_date"], format="%Y%m%d")

        ax.plot(dates, nav_df["nav"], label="Strategy", linewidth=1.5, color="#e74c3c")
        ax.plot(dates, nav_df["bench_nav"], label="CSI 1000", linewidth=1.5, color="#3498db")

        ax.set_title("Strategy NAV vs Benchmark", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        fig.autofmt_xdate()

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "nav_curve.png"), dpi=150)
        plt.close(fig)
        print("  净值曲线已保存: nav_curve.png")

    def _plot_excess_curve(self, nav_df: pd.DataFrame):
        """绘制超额净值曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))

        dates = pd.to_datetime(nav_df["trade_date"], format="%Y%m%d")

        ax.plot(dates, nav_df["excess_nav"], label="Excess NAV", linewidth=1.5, color="#2ecc71")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

        ax.set_title("Excess NAV (Strategy / Benchmark)", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Excess NAV")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        fig.autofmt_xdate()

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "excess_nav_curve.png"), dpi=150)
        plt.close(fig)
        print("  超额净值曲线已保存: excess_nav_curve.png")
