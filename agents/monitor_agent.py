"""
Monitor Agent - factor health monitoring and report generation.
"""
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUT_DIR
from core.agent_base import BaseAgent
from core.signal_bus import Signal
from utils.helpers import ensure_dir
from utils.log import get_logger

logger = get_logger("agents.monitor_agent")


class MonitorAgent(BaseAgent):
    """Monitor Agent: health checks + report output."""

    agent_name = "MonitorAgent"
    IR_THRESHOLD = 0.5

    def _setup_listeners(self):
        # P1: direct event-driven flow factor -> monitor.
        self.listen("factors.tested", self._on_factors_tested)

    def __init__(self):
        super().__init__()
        ensure_dir(OUTPUT_DIR)
        logger.info("初始化完成")

    def _on_factors_tested(self, signal: Signal):
        payload = signal.payload or {}
        factor_test = self._factor_test_from_signal_payload(payload)
        if factor_test is None or factor_test.empty:
            logger.info("factors.tested payload 为空，跳过反馈")
            return
        self.analyze_factors(factor_test)

    @staticmethod
    def _factor_test_from_signal_payload(payload: dict) -> pd.DataFrame | None:
        if "table" in payload and isinstance(payload["table"], pd.DataFrame):
            return payload["table"].copy()

        results = payload.get("results")
        if not isinstance(results, dict) or not results:
            return None

        rows = []
        for name, vals in results.items():
            if not isinstance(vals, dict):
                continue
            rows.append(
                {
                    "factor_name": name,
                    "ic_mean": float(vals.get("ic_mean", 0.0) or 0.0),
                    "ir": float(vals.get("ir", 0.0) or 0.0),
                }
            )
        return pd.DataFrame(rows) if rows else None

    @staticmethod
    def _resolve_factor_columns(factor_test: pd.DataFrame) -> tuple[str, str, str]:
        cols = list(factor_test.columns)

        name_candidates = ["因子名称", "factor_name", "factor", "name"]
        ic_candidates = ["RankIC均值", "ic_mean", "rank_ic_mean", "ic"]
        ir_candidates = ["IR", "ir"]

        def pick(candidates, default_idx):
            for c in candidates:
                if c in cols:
                    return c
            return cols[default_idx]

        name_col = pick(name_candidates, 0)
        ic_col = pick(ic_candidates, 1 if len(cols) > 1 else 0)
        ir_col = pick(ir_candidates, 2 if len(cols) > 2 else 0)
        return name_col, ic_col, ir_col

    def analyze_factors(self, factor_test: pd.DataFrame):
        """Analyze factor health and emit feedback signals for alpha."""
        logger.info("分析因子健康状态...")

        factor_test = factor_test.copy()
        name_col, ic_col, ir_col = self._resolve_factor_columns(factor_test)

        degraded = []
        healthy = {}

        for _, row in factor_test.iterrows():
            fname = str(row[name_col])
            ic_mean = float(row[ic_col]) if pd.notna(row[ic_col]) else 0.0
            ir = float(row[ir_col]) if pd.notna(row[ir_col]) else 0.0

            if abs(ir) < self.IR_THRESHOLD:
                degraded.append(fname)
                logger.info(f"  退化: {fname} |IR|={abs(ir):.4f} < {self.IR_THRESHOLD}")
            else:
                healthy[fname] = abs(ic_mean)
                logger.info(f"  健康: {fname} |IR|={abs(ir):.4f}, |IC|={abs(ic_mean):.4f}")

        if degraded:
            self.emit(
                "monitor.factor_degraded",
                {
                    "degraded_factors": degraded,
                    "reason": f"|IR| < {self.IR_THRESHOLD}",
                },
            )

        if healthy:
            total_ic = sum(healthy.values())
            if total_ic > 0:
                weights = {f: round(ic / total_ic, 4) for f, ic in healthy.items()}
            else:
                weights = {f: round(1.0 / len(healthy), 4) for f in healthy}

            self.emit(
                "monitor.factor_weight_update",
                {
                    "weights": weights,
                    "method": "ic_weighted",
                    "excluded": degraded,
                },
            )
            logger.info(f"IC 加权更新: {weights}")
        else:
            logger.warning("无健康因子，保持等权回退")

    def generate_report(self, metrics: dict, factor_test: pd.DataFrame, nav_df: pd.DataFrame):
        logger.info("生成报告...")
        self._write_text_report(metrics, factor_test)
        self._plot_nav_curve(nav_df)
        self._plot_excess_curve(nav_df)
        logger.info(f"报告已保存至 {OUTPUT_DIR}")

    def _write_text_report(self, metrics: dict, factor_test: pd.DataFrame):
        lines = []
        lines.append("=" * 60)
        lines.append("    Quant Multi-Factor Strategy - Backtest Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append("[1] Strategy Metrics")
        for k, v in metrics.items():
            lines.append(f"  {k:<12}: {v}")
        lines.append("")
        lines.append("[2] Single-Factor Test")
        lines.append(factor_test.to_string(index=False))
        lines.append("")
        lines.append("[3] Notes")
        lines.append("  Universe: CSI1000-like A-share universe")
        lines.append("  Blend: dynamic monitor feedback (IC-weighted), fallback equal-weight")
        lines.append("  Portfolio: Top50 equal-weight")
        lines.append("  Rebalance: Monthly")
        lines.append("")
        lines.append("=" * 60)
        report_text = "\n".join(lines)
        logger.info("\n" + report_text)
        with open(os.path.join(OUTPUT_DIR, "backtest_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_text)

    def _plot_nav_curve(self, nav_df: pd.DataFrame):
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
        logger.info("saved: nav_curve.png")

    def _plot_excess_curve(self, nav_df: pd.DataFrame):
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
        logger.info("saved: excess_nav_curve.png")
