"""端到端迷你流水线冒烟测试"""

import os
import sys
import pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import factors  # noqa: F401
from agents.factor_agent import FactorAgent
from agents.alpha_agent import AlphaAgent
from agents.portfolio_agent import PortfolioAgent
from agents.backtest_agent import BacktestAgent
from utils.helpers import get_month_end_dates


class TestPipelineSmoke:
    """迷你数据端到端冒烟测试"""

    def test_factor_compute(self, mock_daily_quotes):
        """FactorAgent 能在合成数据上计算因子"""
        agent = FactorAgent(mock_daily_quotes)
        result = agent.compute_factors()

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "ts_code" in result.columns
        assert "trade_date" in result.columns
        assert "fwd_ret_1m" in result.columns

    def test_alpha_composite(self, mock_daily_quotes):
        """AlphaAgent 能生成 alpha_score"""
        factor_agent = FactorAgent(mock_daily_quotes)
        factor_agent.compute_factors()

        alpha_agent = AlphaAgent()
        month_ends = get_month_end_dates(factor_agent.factors["trade_date"].unique())

        scores_found = False
        for date in month_ends:
            scores = factor_agent.get_factor_scores(date)
            if len(scores) >= 10:
                alpha_df = alpha_agent.composite_score(scores)
                assert "ts_code" in alpha_df.columns
                assert "alpha_score" in alpha_df.columns
                assert len(alpha_df) > 0
                scores_found = True
                break

        assert scores_found, "没有足够截面数据生成 alpha_score"

    def test_portfolio_select(self, mock_factor_scores):
        """PortfolioAgent 能从因子得分中选股"""
        scores = mock_factor_scores.copy()
        scores["alpha_score"] = scores["factor_a"]

        agent = PortfolioAgent(top_n=10)
        portfolio = agent.select(scores)

        assert len(portfolio) == 10
        assert "ts_code" in portfolio.columns
        assert "weight" in portfolio.columns
        assert abs(portfolio["weight"].sum() - 1.0) < 1e-6

    def test_backtest_runs(self, mock_daily_quotes, mock_benchmark):
        """BacktestAgent 能在合成数据上完成回测"""
        # 构造简单持仓：每期持有前 10 只股票
        codes = sorted(mock_daily_quotes["ts_code"].unique())[:10]
        month_ends = get_month_end_dates(mock_daily_quotes["trade_date"])

        holdings_by_date = {}
        for date in month_ends[:3]:  # 只用 3 期
            holdings_by_date[date] = pd.DataFrame({
                "ts_code": codes,
                "weight": [1.0 / len(codes)] * len(codes),
            })

        agent = BacktestAgent(mock_daily_quotes, mock_benchmark)
        nav_df = agent.run(holdings_by_date)

        assert isinstance(nav_df, pd.DataFrame)
        assert len(nav_df) > 0
        assert "nav" in nav_df.columns
        assert "bench_nav" in nav_df.columns

    def test_calc_metrics(self, mock_daily_quotes, mock_benchmark):
        """calc_metrics 返回预期的指标 key"""
        codes = sorted(mock_daily_quotes["ts_code"].unique())[:10]
        month_ends = get_month_end_dates(mock_daily_quotes["trade_date"])

        holdings_by_date = {}
        for date in month_ends[:3]:
            holdings_by_date[date] = pd.DataFrame({
                "ts_code": codes,
                "weight": [1.0 / len(codes)] * len(codes),
            })

        agent = BacktestAgent(mock_daily_quotes, mock_benchmark)
        nav_df = agent.run(holdings_by_date)
        metrics = BacktestAgent.calc_metrics(nav_df)

        expected_keys = ["总收益率", "年化收益率", "夏普比率", "最大回撤"]
        for key in expected_keys:
            assert key in metrics, f"缺少指标: {key}"
