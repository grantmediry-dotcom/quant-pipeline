"""
Portfolio Agent - 组合构建
支持等权、均值方差优化（MVO）、风险平价三种模式
"""
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    TOP_N, PORTFOLIO_METHOD, MVO_MAX_WEIGHT, MVO_RISK_AVERSION, RISK_PARITY_MAX_WEIGHT,
)
from core.agent_base import BaseAgent
from utils.log import get_logger

logger = get_logger("agents.portfolio_agent")


class PortfolioAgent(BaseAgent):
    """组合Agent：支持等权/MVO/风险平价"""

    agent_name = "PortfolioAgent"

    def __init__(self, top_n: int = TOP_N, daily_quotes: pd.DataFrame = None):
        super().__init__()
        self.top_n = top_n
        self.daily_quotes = daily_quotes
        logger.info(f"初始化完成，每期选 {top_n} 只，模式: {PORTFOLIO_METHOD}")

    def select(self, alpha_scores: pd.DataFrame) -> pd.DataFrame:
        """
        根据 alpha_score 选股并分配权重

        输入：含 ts_code, alpha_score 的 DataFrame
        输出：含 ts_code, weight 的 DataFrame
        """
        if PORTFOLIO_METHOD == "mvo":
            return self._mvo(alpha_scores)
        elif PORTFOLIO_METHOD == "risk_parity":
            return self._risk_parity(alpha_scores)
        else:
            return self._equal_weight(alpha_scores)

    def _equal_weight(self, alpha_scores: pd.DataFrame) -> pd.DataFrame:
        """等权配置"""
        df = alpha_scores.sort_values("alpha_score", ascending=False).head(self.top_n)
        df = df.copy()
        df["weight"] = 1.0 / len(df) if len(df) > 0 else 0
        return df[["ts_code", "weight"]]

    def _mvo(self, alpha_scores: pd.DataFrame) -> pd.DataFrame:
        """均值-方差优化"""
        from optimization.mvo import MeanVarianceOptimizer

        # 预筛选 Top 2N 候选（减少优化维度）
        candidates = alpha_scores.sort_values("alpha_score", ascending=False).head(self.top_n * 2)
        codes = candidates["ts_code"].tolist()

        if len(codes) < 5:
            return self._equal_weight(alpha_scores)

        optimizer = MeanVarianceOptimizer()

        # 估算协方差矩阵
        if self.daily_quotes is not None:
            cov = optimizer.estimate_covariance(self.daily_quotes, codes)
        else:
            cov = np.eye(len(codes)) * 0.04 / 252

        # alpha_score 作为预期收益信号
        alpha_series = candidates.set_index("ts_code")["alpha_score"]

        weights = optimizer.optimize(
            alpha_series, cov,
            max_weight=MVO_MAX_WEIGHT,
            risk_aversion=MVO_RISK_AVERSION,
        )

        # 筛掉零权重
        weights = weights[weights > 0]
        if weights.empty:
            logger.warning("MVO 优化无有效权重，回退等权")
            return self._equal_weight(alpha_scores)

        result = pd.DataFrame({
            "ts_code": weights.index,
            "weight": weights.values,
        })
        return result

    def _risk_parity(self, alpha_scores: pd.DataFrame) -> pd.DataFrame:
        """风险平价优化"""
        from optimization.risk_parity import RiskParityOptimizer
        from optimization.mvo import MeanVarianceOptimizer

        # 预筛选 Top N（风险平价不需要过多候选）
        candidates = alpha_scores.sort_values("alpha_score", ascending=False).head(self.top_n)
        codes = candidates["ts_code"].tolist()

        if len(codes) < 5:
            return self._equal_weight(alpha_scores)

        # 估算协方差
        if self.daily_quotes is not None:
            cov = MeanVarianceOptimizer.estimate_covariance(self.daily_quotes, codes)
        else:
            cov = np.eye(len(codes)) * 0.04 / 252

        optimizer = RiskParityOptimizer()
        weights = optimizer.optimize(cov, max_weight=RISK_PARITY_MAX_WEIGHT)

        result = pd.DataFrame({
            "ts_code": codes[:len(weights)],
            "weight": weights,
        })
        result = result[result["weight"] > 0].copy()

        if result.empty:
            logger.warning("风险平价优化无有效权重，回退等权")
            return self._equal_weight(alpha_scores)

        return result[["ts_code", "weight"]]
