"""
Orchestrator — 总指挥（V3）

替代 main.py 的硬编码线性流程，引入反馈闭环：
  Phase 1: 数据准备
  Phase 2: 因子计算 + 检验
  Phase 3: 监控反馈（MonitorAgent 分析因子→发出权重信号）
  Phase 4: Alpha 合成 + 选股（使用调整后权重）
  Phase 5: 回测
  Phase 6: 报告 + 信号日志

关键：Phase 3 在 Phase 4 之前，所以因子权重调整在合成之前生效。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.signal_bus import SignalBus
from agents.data_agent import DataAgent
from agents.factor_agent import FactorAgent
from agents.alpha_agent import AlphaAgent
from agents.portfolio_agent import PortfolioAgent
from agents.backtest_agent import BacktestAgent
from agents.monitor_agent import MonitorAgent


class Orchestrator:
    """
    流程总指挥

    职责：
    1. 初始化信号总线
    2. 按阶段调度 Agent 执行
    3. 在 Phase 3 和 Phase 4 之间中继信号（解决时序问题）
    4. 打印信号日志
    """

    def __init__(self):
        self.bus = SignalBus()
        self.bus.clear()

    def run(self):
        print("=" * 60)
        print("  量化多因子策略 V3 — Agent 通信版")
        print("  Data → Factor → [Monitor 反馈] → Alpha → Portfolio → Backtest")
        print("=" * 60)

        # ===== Phase 1: 数据 =====
        print("\n--- Phase 1: 数据准备 ---")
        data_agent = DataAgent()
        data_agent.update_all()
        daily_quotes = data_agent.get_daily_quotes()
        benchmark = data_agent.get_benchmark()

        # ===== Phase 2: 因子计算 + 检验 =====
        print("\n--- Phase 2: 因子计算与检验 ---")
        factor_agent = FactorAgent(daily_quotes)
        factor_agent.compute_factors()
        factor_test_result = factor_agent.single_factor_test()

        # ===== Phase 3: 监控反馈（关键：在 Alpha 之前） =====
        print("\n--- Phase 3: 监控反馈分析 ---")
        monitor_agent = MonitorAgent()
        monitor_agent.analyze_factors(factor_test_result)

        # ===== Phase 4: Alpha 合成 + 选股 =====
        print("\n--- Phase 4: Alpha 合成 + 选股 ---")
        alpha_agent = AlphaAgent()

        # Orchestrator 中继：从信号历史中提取权重更新，传给 AlphaAgent
        weight_update = self._get_latest_signal("monitor.factor_weight_update")
        if weight_update:
            alpha_agent.set_factor_weights(
                weight_update["weights"],
                weight_update.get("excluded", []),
            )

        portfolio_agent = PortfolioAgent()

        factors_df = factor_agent.factors.copy()
        factors_df["month"] = factors_df["trade_date"].str[:6]
        month_ends = factors_df.groupby("month")["trade_date"].max().values

        # 预计算流动性过滤：20日均成交额 < 1000万元(amount单位千元,阈值10000)
        daily_quotes["_avg_amount_20d"] = daily_quotes.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )

        holdings_by_date = {}
        for date in sorted(month_ends):
            scores = factor_agent.get_factor_scores(date)
            if len(scores) < 50:
                continue
            alpha_scores = alpha_agent.composite_score(scores)

            # 过滤流动性不足的股票（日均成交额 < 1000万元）
            illiquid_codes = daily_quotes[
                (daily_quotes["trade_date"] == date) &
                (daily_quotes["_avg_amount_20d"] < 10000)
            ]["ts_code"].tolist()
            if illiquid_codes:
                alpha_scores = alpha_scores[
                    ~alpha_scores["ts_code"].isin(illiquid_codes)
                ]

            # 过滤涨停股票（涨停无法买入）
            if "is_limit_up" in daily_quotes.columns:
                limit_up_codes = daily_quotes[
                    (daily_quotes["trade_date"] == date) &
                    (daily_quotes["is_limit_up"] == 1)
                ]["ts_code"].tolist()
                if limit_up_codes:
                    alpha_scores = alpha_scores[
                        ~alpha_scores["ts_code"].isin(limit_up_codes)
                    ]

            portfolio = portfolio_agent.select(alpha_scores)
            holdings_by_date[date] = portfolio

        print(f"\n[Orchestrator] 共生成 {len(holdings_by_date)} 期持仓")

        # ===== Phase 5: 回测 =====
        print("\n--- Phase 5: 回测 ---")
        backtest_agent = BacktestAgent(daily_quotes, benchmark)
        nav_df = backtest_agent.run(holdings_by_date)
        metrics = BacktestAgent.calc_metrics(nav_df)

        # ===== Phase 6: 报告 =====
        print("\n--- Phase 6: 报告生成 ---")
        monitor_agent.generate_report(metrics, factor_test_result, nav_df)

        # ===== 信号日志 =====
        print("\n" + "=" * 60)
        print("  信号通信日志")
        print("=" * 60)
        print(self.bus.summary())
        for sig in self.bus.get_history():
            payload_keys = list(sig.payload.keys())
            print(f"  [{sig.timestamp}] {sig.sender} -> {sig.name} ({payload_keys})")

        print("\n" + "=" * 60)
        print("  V3 全流程执行完成！")
        print(f"  报告输出目录: {os.path.join(PROJECT_ROOT, 'output')}")
        print("=" * 60)

    def _get_latest_signal(self, signal_name: str) -> dict | None:
        """从信号历史中获取最新的指定信号 payload"""
        for signal in reversed(self.bus.get_history()):
            if signal.name == signal_name:
                return signal.payload
        return None
