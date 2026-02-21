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
        portfolio_agent = PortfolioAgent()

        month_ends = get_month_end_dates(factor_agent.factors["trade_date"].unique())

        # Liquidity filter: 20-day avg amount < 10m CNY (amount unit: thousand CNY)
        daily_quotes["_avg_amount_20d"] = calc_avg_amount(daily_quotes)

        holdings_by_date = {}
        for date in sorted(month_ends):
            scores = factor_agent.get_factor_scores(date)
            if len(scores) < 50:
                continue
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

        # Phase 5: Report
        logger.info("--- Phase 5: Report ---")
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
