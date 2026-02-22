"""
Orchestrator - V3 workflow controller.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.alpha_agent import AlphaAgent
from agents.backtest_agent import BacktestAgent
from agents.data_agent import DataAgent
from agents.factor_agent import FactorAgent
from agents.monitor_agent import MonitorAgent
from agents.portfolio_agent import PortfolioAgent
from core.signal_bus import SignalBus
from utils.helpers import get_month_end_dates, calc_avg_amount, get_illiquid_codes
from utils.log import get_logger
from config.settings import OUTPUT_DIR, REGIME_DETECTION_ENABLED

logger = get_logger("core.orchestrator")


class Orchestrator:
    """Workflow controller for end-to-end strategy pipeline."""

    def __init__(self):
        self.bus = SignalBus()
        self.bus.clear()

    def run(self):
        logger.info("=" * 60)
        logger.info("  Quant Multi-Factor Strategy V3 - Agent Event Mode")
        logger.info("  Data -> Factor -> Monitor -> Alpha -> Portfolio -> Backtest")
        logger.info("=" * 60)

        # Phase 1: Data
        logger.info("--- Phase 1: Data ---")
        data_agent = DataAgent()
        data_agent.update_all()
        daily_quotes = data_agent.get_daily_quotes()
        benchmark = data_agent.get_benchmark()

        # Phase 2: Factor + event-driven monitor feedback
        logger.info("--- Phase 2: Factor + Monitor Feedback ---")
        factor_agent = FactorAgent(daily_quotes)
        monitor_agent = MonitorAgent()  # subscribe before factor signal
        alpha_agent = AlphaAgent()      # subscribe before monitor signal

        factor_agent.compute_factors()
        factor_test_result = factor_agent.single_factor_test()

        # Phase 3: Alpha + Portfolio
        logger.info("--- Phase 3: Alpha + Portfolio ---")
        # Liquidity filter: 20-day avg amount < 10m CNY (amount unit: thousand CNY)
        daily_quotes["_avg_amount_20d"] = calc_avg_amount(daily_quotes)

        portfolio_agent = PortfolioAgent(daily_quotes=daily_quotes)

        month_ends = get_month_end_dates(factor_agent.factors["trade_date"].unique())

        # 状态识别（P2d，可选）
        regime_detector = None
        regime_history = None
        if REGIME_DETECTION_ENABLED:
            from analytics.regime import RegimeDetector
            regime_detector = RegimeDetector()
            regime_history = regime_detector.get_history(benchmark)
            regime_detector.plot_regime_overlay(benchmark, regime_history, OUTPUT_DIR)
            logger.info("市场状态识别已启用")

        holdings_by_date = {}
        factor_scores_history = {}
        for date in sorted(month_ends):
            scores = factor_agent.get_factor_scores(date)
            if len(scores) < 50:
                continue
            factor_scores_history[date] = scores.copy()

            # 状态驱动权重调整
            if regime_detector is not None:
                regime = regime_detector.detect(benchmark, date)
                if regime == "bear":
                    # 熊市：降低动量权重，增加低波动权重
                    alpha_agent.regime_adjust = {"momentum_20d": 0.5, "volatility_20d": 1.5}
                elif regime == "bull":
                    alpha_agent.regime_adjust = {"momentum_20d": 1.2}
                else:
                    alpha_agent.regime_adjust = {}

            alpha_scores = alpha_agent.composite_score(scores)

            illiquid_codes = get_illiquid_codes(daily_quotes, date)
            if illiquid_codes:
                alpha_scores = alpha_scores[~alpha_scores["ts_code"].isin(illiquid_codes)]

            if "is_limit_up" in daily_quotes.columns:
                limit_up_codes = daily_quotes[
                    (daily_quotes["trade_date"] == date)
                    & (daily_quotes["is_limit_up"] == 1)
                ]["ts_code"].tolist()
                if limit_up_codes:
                    alpha_scores = alpha_scores[~alpha_scores["ts_code"].isin(limit_up_codes)]

            portfolio = portfolio_agent.select(alpha_scores)
            holdings_by_date[date] = portfolio

        logger.info(f"generated holdings periods: {len(holdings_by_date)}")

        # Phase 4: Backtest
        logger.info("--- Phase 4: Backtest ---")
        backtest_agent = BacktestAgent(daily_quotes, benchmark)
        nav_df = backtest_agent.run(holdings_by_date)
        metrics = BacktestAgent.calc_metrics(nav_df)

        # Phase 5: Attribution
        logger.info("--- Phase 5: Attribution ---")
        from analytics.attribution import PerformanceAttributor
        attributor = PerformanceAttributor()
        attr_data = backtest_agent.get_attribution_data()
        factor_weights = alpha_agent.factor_weights or {}
        attribution_df = attributor.factor_pnl_decomposition(
            attr_data["holdings_history"], factor_scores_history,
            factor_weights, daily_quotes,
        )
        attributor.save_report(attribution_df, nav_df, OUTPUT_DIR)

        # Phase 6: Report
        logger.info("--- Phase 6: Report ---")
        monitor_agent.generate_report(metrics, factor_test_result, nav_df)

        # Signal logs
        logger.info("=" * 60)
        logger.info("  Signal Log")
        logger.info("=" * 60)
        logger.info(self.bus.summary())
        for sig in self.bus.get_history():
            payload_keys = list(sig.payload.keys())
            logger.info(f"  [{sig.timestamp}] {sig.sender} -> {sig.name} ({payload_keys})")

        logger.info("=" * 60)
        logger.info("  V3 workflow completed")
        logger.info(f"  output dir: {os.path.join(PROJECT_ROOT, 'output')}")
        logger.info("=" * 60)
