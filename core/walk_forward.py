"""
Walk-Forward 样本外验证

核心思想：在训练窗口计算因子IC/IR和权重，在测试窗口验证策略表现。
滚动前进，拼接所有测试窗口的NAV曲线，得到真实的样本外绩效。

用法：
    python main_walkforward.py

默认窗口：2年训练 + 6个月测试，步进6个月
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.signal_bus import SignalBus
from agents.data_agent import DataAgent
from agents.factor_agent import FactorAgent
from agents.alpha_agent import AlphaAgent
from agents.portfolio_agent import PortfolioAgent
from agents.backtest_agent import BacktestAgent
from agents.monitor_agent import MonitorAgent
from factor_framework.registry import FactorRegistry
from config.settings import OUTPUT_DIR, START_DATE, END_DATE
from utils.helpers import ensure_dir, get_month_end_dates, calc_avg_amount, get_illiquid_codes
from utils.log import get_logger

logger = get_logger("core.walk_forward")


def generate_windows(
    start: str = START_DATE,
    end: str = END_DATE,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> list[dict]:
    """
    生成 walk-forward 滚动窗口列表

    返回：[{"train": (start, end), "test": (start, end)}, ...]
    """
    from dateutil.relativedelta import relativedelta

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    windows = []
    train_start = start_dt

    while True:
        train_end = train_start + relativedelta(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - pd.Timedelta(days=1)

        if test_end > end_dt:
            test_end = end_dt

        if test_start > end_dt:
            break

        windows.append({
            "train": (train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d")),
            "test": (test_start.strftime("%Y%m%d"), test_end.strftime("%Y%m%d")),
        })

        train_start = train_start + relativedelta(months=step_months)

        if test_end >= end_dt:
            break

    return windows


class WalkForwardValidator:
    """
    Walk-Forward 样本外验证器

    流程：
    1. 一次性加载数据和计算因子（全量，保证lookback完整）
    2. 对每个窗口：
       a. 训练期：计算IC/IR → Monitor评估 → 生成IC加权权重
       b. 测试期：用训练期权重 + 测试期因子得分 → 选股 → 回测
    3. 拼接所有测试期NAV → 输出样本外绩效
    """

    def __init__(self, windows: list[dict] = None):
        self.bus = SignalBus()
        self.windows = windows or generate_windows()
        self.registry = FactorRegistry()
        ensure_dir(OUTPUT_DIR)

    def run(self):
        logger.info("=" * 60)
        logger.info("  Walk-Forward 样本外验证")
        logger.info(f"  共 {len(self.windows)} 个滚动窗口")
        logger.info("=" * 60)

        # ===== Phase 1: 一次性数据准备 =====
        logger.info("--- 数据准备 ---")
        data_agent = DataAgent()
        data_agent.update_all()
        daily_quotes = data_agent.get_daily_quotes()
        benchmark = data_agent.get_benchmark()

        # ===== Phase 2: 一次性因子计算（全量数据） =====
        logger.info("--- 因子计算（全量） ---")
        factor_agent = FactorAgent(daily_quotes)
        factor_agent.compute_factors()

        # 预计算流动性过滤
        daily_quotes["_avg_amount_20d"] = calc_avg_amount(daily_quotes)

        # 获取所有月末日期
        all_month_ends = get_month_end_dates(factor_agent.factors["trade_date"].unique())

        # ===== Phase 3: 逐窗口执行 =====
        window_results = []
        window_summaries = []

        for i, window in enumerate(self.windows):
            train_range = window["train"]
            test_range = window["test"]

            logger.info("=" * 60)
            logger.info(f"  Window {i+1}/{len(self.windows)}")
            logger.info(f"  Train: {train_range[0]} ~ {train_range[1]}")
            logger.info(f"  Test:  {test_range[0]} ~ {test_range[1]}")
            logger.info("=" * 60)

            result = self._run_single_window(
                factor_agent=factor_agent,
                daily_quotes=daily_quotes,
                benchmark=benchmark,
                all_month_ends=all_month_ends,
                train_range=train_range,
                test_range=test_range,
                window_idx=i,
            )

            if result is not None:
                window_results.append(result)
                window_summaries.append(result["summary"])

        if not window_results:
            logger.warning("没有有效的测试窗口结果")
            return

        # ===== Phase 4: 拼接 + 报告 =====
        logger.info("=" * 60)
        logger.info("  拼接样本外结果")
        logger.info("=" * 60)

        oos_nav = self._stitch_nav(window_results, benchmark)
        oos_metrics = BacktestAgent.calc_metrics(oos_nav)

        # 全量回测对比（样本内）
        logger.info("--- 全量回测（样本内对比） ---")
        is_nav = self._run_full_period(
            factor_agent, daily_quotes, benchmark, all_month_ends
        )
        is_metrics = BacktestAgent.calc_metrics(is_nav) if is_nav is not None else {}

        # 输出
        self._save_results(oos_nav, oos_metrics, is_metrics, window_summaries)

        logger.info("=" * 60)
        logger.info("  Walk-Forward 验证完成！")
        logger.info(f"  输出目录: {OUTPUT_DIR}")
        logger.info("=" * 60)

    def _run_single_window(
        self,
        factor_agent: FactorAgent,
        daily_quotes: pd.DataFrame,
        benchmark: pd.DataFrame,
        all_month_ends: list,
        train_range: tuple[str, str],
        test_range: tuple[str, str],
        window_idx: int,
    ) -> dict | None:
        """执行单个 walk-forward 窗口"""

        # 1. 重置信号总线 + 因子评估指标
        self.bus.clear()
        self.registry.reset_eval()

        # 2. 训练期：因子检验（窗口内IC/IR）
        factor_test = factor_agent.single_factor_test(date_range=train_range)

        if factor_test.empty:
            logger.warning(f"[Window {window_idx+1}] 训练期无有效因子检验结果，跳过")
            return None

        # 3. Monitor 分析训练期因子健康状态
        monitor = MonitorAgent()
        monitor.analyze_factors(factor_test)

        # 4. Alpha 接收训练期权重
        alpha_agent = AlphaAgent()
        weight_signal = self._get_latest_signal("monitor.factor_weight_update")
        if weight_signal:
            alpha_agent.set_factor_weights(
                weight_signal["weights"],
                weight_signal.get("excluded", []),
            )
            active_weights = weight_signal["weights"]
        else:
            active_weights = {}

        # 5. 测试期：选股
        portfolio_agent = PortfolioAgent(daily_quotes=daily_quotes)
        test_month_ends = [
            d for d in all_month_ends
            if d > train_range[1] and d <= test_range[1]
        ]

        if not test_month_ends:
            logger.warning(f"[Window {window_idx+1}] 测试期无调仓日，跳过")
            return None

        holdings_by_date = {}
        for date in test_month_ends:
            scores = factor_agent.get_factor_scores(date)
            if len(scores) < 50:
                continue

            alpha_scores = alpha_agent.composite_score(scores)

            # 流动性过滤
            illiquid = get_illiquid_codes(daily_quotes, date)
            if illiquid:
                alpha_scores = alpha_scores[~alpha_scores["ts_code"].isin(illiquid)]

            # 涨停过滤
            if "is_limit_up" in daily_quotes.columns:
                limit_up = daily_quotes[
                    (daily_quotes["trade_date"] == date) &
                    (daily_quotes["is_limit_up"] == 1)
                ]["ts_code"].tolist()
                if limit_up:
                    alpha_scores = alpha_scores[~alpha_scores["ts_code"].isin(limit_up)]

            portfolio = portfolio_agent.select(alpha_scores)
            holdings_by_date[date] = portfolio

        if not holdings_by_date:
            logger.warning(f"[Window {window_idx+1}] 测试期无有效持仓，跳过")
            return None

        logger.info(f"[Window {window_idx+1}] 测试期生成 {len(holdings_by_date)} 期持仓")

        # 6. 测试期回测
        backtest_agent = BacktestAgent(daily_quotes, benchmark)
        nav_df = backtest_agent.run(holdings_by_date)

        # 仅保留测试期的 NAV
        nav_df = nav_df[
            (nav_df["trade_date"] >= test_range[0]) &
            (nav_df["trade_date"] <= test_range[1])
        ].copy()

        if nav_df.empty:
            return None

        # 窗口汇总
        test_metrics = BacktestAgent.calc_metrics(nav_df) if len(nav_df) > 10 else {}

        # 训练期因子指标
        train_ir = {row["因子名称"]: row["IR"] for _, row in factor_test.iterrows()}

        summary = {
            "window": window_idx + 1,
            "train_start": train_range[0],
            "train_end": train_range[1],
            "test_start": test_range[0],
            "test_end": test_range[1],
            "n_holdings_periods": len(holdings_by_date),
            "active_factors": len(active_weights),
            "weights": active_weights,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

        logger.info(f"[Window {window_idx+1}] 测试期绩效: {test_metrics.get('年化收益率', 'N/A')}")

        return {
            "nav_df": nav_df,
            "factor_test": factor_test,
            "summary": summary,
        }

    def _run_full_period(self, factor_agent, daily_quotes, benchmark, all_month_ends):
        """全量回测（用于样本内对比）"""
        self.bus.clear()
        self.registry.reset_eval()

        factor_test = factor_agent.single_factor_test()
        monitor = MonitorAgent()
        monitor.analyze_factors(factor_test)

        alpha_agent = AlphaAgent()
        weight_signal = self._get_latest_signal("monitor.factor_weight_update")
        if weight_signal:
            alpha_agent.set_factor_weights(
                weight_signal["weights"], weight_signal.get("excluded", [])
            )

        portfolio_agent = PortfolioAgent(daily_quotes=daily_quotes)
        holdings_by_date = {}
        for date in all_month_ends:
            scores = factor_agent.get_factor_scores(date)
            if len(scores) < 50:
                continue
            alpha_scores = alpha_agent.composite_score(scores)

            illiquid = get_illiquid_codes(daily_quotes, date)
            if illiquid:
                alpha_scores = alpha_scores[~alpha_scores["ts_code"].isin(illiquid)]

            portfolio = portfolio_agent.select(alpha_scores)
            holdings_by_date[date] = portfolio

        backtest_agent = BacktestAgent(daily_quotes, benchmark)
        return backtest_agent.run(holdings_by_date)

    def _stitch_nav(self, window_results: list, benchmark: pd.DataFrame) -> pd.DataFrame:
        """拼接所有测试窗口的 NAV 曲线"""
        cumulative_nav = 1.0
        stitched = []

        for result in window_results:
            nav_df = result["nav_df"].copy()
            if nav_df.empty:
                continue

            # 归一化：每段起点接上上一段终点
            start_nav = nav_df["nav"].iloc[0]
            if start_nav > 0:
                nav_df["nav"] = nav_df["nav"] / start_nav * cumulative_nav

            # 同步归一化基准
            start_bench = nav_df["bench_nav"].iloc[0]
            if start_bench > 0:
                nav_df["bench_nav"] = nav_df["bench_nav"] / start_bench

            cumulative_nav = nav_df["nav"].iloc[-1]
            stitched.append(nav_df)

        if not stitched:
            return pd.DataFrame()

        combined = pd.concat(stitched, ignore_index=True)
        combined = combined.drop_duplicates(subset="trade_date", keep="last")
        combined = combined.sort_values("trade_date").reset_index(drop=True)

        # 重新计算基准NAV（从拼接后的完整基准收盘价）
        combined = combined.merge(
            benchmark[["trade_date", "bench_close"]],
            on="trade_date", how="left", suffixes=("_old", "")
        )
        if "bench_close_old" in combined.columns:
            combined = combined.drop(columns=["bench_close_old"])
        combined["bench_close"] = combined["bench_close"].ffill()
        if combined["bench_close"].iloc[0] > 0:
            combined["bench_nav"] = combined["bench_close"] / combined["bench_close"].iloc[0]
        combined["excess_nav"] = combined["nav"] / combined["bench_nav"]

        return combined

    def _get_latest_signal(self, signal_name: str) -> dict | None:
        """从信号历史中获取最新的指定信号"""
        for signal in reversed(self.bus.get_history()):
            if signal.name == signal_name:
                return signal.payload
        return None

    def _save_results(
        self,
        oos_nav: pd.DataFrame,
        oos_metrics: dict,
        is_metrics: dict,
        window_summaries: list,
    ):
        """保存所有结果"""
        ensure_dir(OUTPUT_DIR)

        # 1. NAV CSV
        oos_nav.to_csv(
            os.path.join(OUTPUT_DIR, "walkforward_nav.csv"),
            index=False, encoding="utf-8-sig"
        )

        # 2. 窗口汇总
        summary_df = pd.DataFrame(window_summaries)
        summary_df.to_csv(
            os.path.join(OUTPUT_DIR, "walkforward_window_summary.csv"),
            index=False, encoding="utf-8-sig"
        )

        # 3. 文字报告
        self._write_report(oos_metrics, is_metrics, window_summaries)

        # 4. 净值曲线图
        self._plot_nav(oos_nav)
        self._plot_comparison(oos_nav)

    def _write_report(self, oos_metrics, is_metrics, window_summaries):
        """生成 walk-forward 报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("    Walk-Forward 样本外验证报告")
        lines.append("=" * 60)
        lines.append("")

        lines.append("【一、样本外绩效 (OOS)】")
        for k, v in oos_metrics.items():
            lines.append(f"  {k}: {v}")

        lines.append("")
        lines.append("【二、样本内绩效 (IS) — 全量回测对比】")
        for k, v in is_metrics.items():
            lines.append(f"  {k}: {v}")

        lines.append("")
        lines.append("【三、IS/OOS 衰减分析】")
        for key in ["年化收益率", "年化超额收益", "夏普比率"]:
            is_val = is_metrics.get(key, "N/A")
            oos_val = oos_metrics.get(key, "N/A")
            lines.append(f"  {key}: IS={is_val}  OOS={oos_val}")

        lines.append("")
        lines.append("【四、各窗口详情】")
        for s in window_summaries:
            lines.append(f"  Window {s['window']}: "
                         f"Train {s['train_start']}~{s['train_end']} | "
                         f"Test {s['test_start']}~{s['test_end']} | "
                         f"活跃因子 {s['active_factors']} 个 | "
                         f"持仓期 {s['n_holdings_periods']} 期")
            if s.get("weights"):
                for f, w in s["weights"].items():
                    lines.append(f"    {f}: {w:.1%}")

        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)
        logger.info("\n" + report)

        with open(os.path.join(OUTPUT_DIR, "walkforward_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    def _plot_nav(self, nav_df: pd.DataFrame):
        """样本外净值曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = pd.to_datetime(nav_df["trade_date"], format="%Y%m%d")

        ax.plot(dates, nav_df["nav"], label="Strategy (OOS)", linewidth=1.5, color="#e74c3c")
        ax.plot(dates, nav_df["bench_nav"], label="CSI 1000", linewidth=1.5, color="#3498db")

        ax.set_title("Walk-Forward Out-of-Sample NAV", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "walkforward_nav_curve.png"), dpi=150)
        plt.close(fig)
        logger.info("样本外净值曲线已保存: walkforward_nav_curve.png")

    def _plot_comparison(self, oos_nav: pd.DataFrame):
        """超额净值曲线"""
        if "excess_nav" not in oos_nav.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        dates = pd.to_datetime(oos_nav["trade_date"], format="%Y%m%d")

        ax.plot(dates, oos_nav["excess_nav"], label="Excess NAV (OOS)",
                linewidth=1.5, color="#2ecc71")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

        ax.set_title("Walk-Forward Excess NAV (OOS)", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Excess NAV")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "walkforward_excess_nav.png"), dpi=150)
        plt.close(fig)
        logger.info("样本外超额净值曲线已保存: walkforward_excess_nav.png")
