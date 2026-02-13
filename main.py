"""
量化多因子策略 — 最小闭环主流程
Data Agent → Factor Agent → Alpha Agent → Portfolio Agent → Backtest Agent → Monitor Agent
"""
import os
import sys

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from agents.data_agent import DataAgent
from agents.factor_agent import FactorAgent
from agents.alpha_agent import AlphaAgent
from agents.portfolio_agent import PortfolioAgent
from agents.backtest_agent import BacktestAgent
from agents.monitor_agent import MonitorAgent


def main():
    print("=" * 60)
    print("  量化多因子策略 — 最小闭环")
    print("  Data → Factor → Alpha → Portfolio → Backtest → Monitor")
    print("=" * 60)
    print()

    # ===== 1. Data Agent: 拉取数据 =====
    data_agent = DataAgent()
    data_agent.update_all()
    daily_quotes = data_agent.get_daily_quotes()
    benchmark = data_agent.get_benchmark()

    # ===== 2. Factor Agent: 计算因子 + 单因子检验 =====
    factor_agent = FactorAgent(daily_quotes)
    factor_agent.compute_factors()
    factor_test_result = factor_agent.single_factor_test()

    # ===== 3 & 4. Alpha + Portfolio: 生成每期持仓 =====
    alpha_agent = AlphaAgent()
    portfolio_agent = PortfolioAgent()

    # 获取月末调仓日列表
    factors_df = factor_agent.factors
    factors_df["month"] = factors_df["trade_date"].str[:6]
    month_ends = factors_df.groupby("month")["trade_date"].max().values

    holdings_by_date = {}
    for date in sorted(month_ends):
        # 获取截面因子得分
        scores = factor_agent.get_factor_scores(date)
        if len(scores) < 50:
            continue

        # 合成 Alpha
        alpha_scores = alpha_agent.composite_score(scores)

        # 选股
        portfolio = portfolio_agent.select(alpha_scores)
        holdings_by_date[date] = portfolio

    print(f"\n[Main] 共生成 {len(holdings_by_date)} 期持仓")

    # ===== 5. Backtest Agent: 回测 =====
    backtest_agent = BacktestAgent(daily_quotes, benchmark)
    nav_df = backtest_agent.run(holdings_by_date)
    metrics = BacktestAgent.calc_metrics(nav_df)

    # ===== 6. Monitor Agent: 生成报告 =====
    monitor_agent = MonitorAgent()
    monitor_agent.generate_report(metrics, factor_test_result, nav_df)

    print("\n" + "=" * 60)
    print("  全流程执行完成！")
    print(f"  报告输出目录: {os.path.join(PROJECT_ROOT, 'output')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
