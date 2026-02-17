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


class Orchestrator:
    """Workflow controller for end-to-end strategy pipeline."""

    def __init__(self):
        self.bus = SignalBus()
        self.bus.clear()

    def run(self):
        print("=" * 60)
        print("  Quant Multi-Factor Strategy V3 - Agent Event Mode")
        print("  Data -> Factor -> Monitor -> Alpha -> Portfolio -> Backtest")
        print("=" * 60)

        # Phase 1: Data
        print("\n--- Phase 1: Data ---")
        data_agent = DataAgent()
        data_agent.update_all()
        daily_quotes = data_agent.get_daily_quotes()
        benchmark = data_agent.get_benchmark()

        # Phase 2: Factor + event-driven monitor feedback
        print("\n--- Phase 2: Factor + Monitor Feedback ---")
        factor_agent = FactorAgent(daily_quotes)
        monitor_agent = MonitorAgent()  # subscribe before factor signal
        alpha_agent = AlphaAgent()      # subscribe before monitor signal

        factor_agent.compute_factors()
        factor_test_result = factor_agent.single_factor_test()

        # Phase 3: Alpha + Portfolio
        print("\n--- Phase 3: Alpha + Portfolio ---")
        portfolio_agent = PortfolioAgent()

        factors_df = factor_agent.factors.copy()
        factors_df["month"] = factors_df["trade_date"].str[:6]
        month_ends = factors_df.groupby("month")["trade_date"].max().values

        # Liquidity filter: 20-day avg amount < 10m CNY (amount unit: thousand CNY)
        daily_quotes["_avg_amount_20d"] = daily_quotes.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )

        holdings_by_date = {}
        for date in sorted(month_ends):
            scores = factor_agent.get_factor_scores(date)
            if len(scores) < 50:
                continue
            alpha_scores = alpha_agent.composite_score(scores)

            illiquid_codes = daily_quotes[
                (daily_quotes["trade_date"] == date)
                & (daily_quotes["_avg_amount_20d"] < 10000)
            ]["ts_code"].tolist()
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

        print(f"\n[Orchestrator] generated holdings periods: {len(holdings_by_date)}")

        # Phase 4: Backtest
        print("\n--- Phase 4: Backtest ---")
        backtest_agent = BacktestAgent(daily_quotes, benchmark)
        nav_df = backtest_agent.run(holdings_by_date)
        metrics = BacktestAgent.calc_metrics(nav_df)

        # Phase 5: Report
        print("\n--- Phase 5: Report ---")
        monitor_agent.generate_report(metrics, factor_test_result, nav_df)

        # Signal logs
        print("\n" + "=" * 60)
        print("  Signal Log")
        print("=" * 60)
        print(self.bus.summary())
        for sig in self.bus.get_history():
            payload_keys = list(sig.payload.keys())
            print(f"  [{sig.timestamp}] {sig.sender} -> {sig.name} ({payload_keys})")

        print("\n" + "=" * 60)
        print("  V3 workflow completed")
        print(f"  output dir: {os.path.join(PROJECT_ROOT, 'output')}")
        print("=" * 60)
