"""
风险平价优化 (Risk Parity)

目标: 最小化各资产风险贡献的方差，使每个资产的风险贡献相等。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParityOptimizer:
    """风险平价组合优化器"""

    def optimize(
        self,
        cov_matrix: np.ndarray,
        max_weight: float = 0.15,
    ) -> np.ndarray:
        """
        风险平价优化

        参数:
            cov_matrix: n×n 协方差矩阵
            max_weight: 单只资产最大权重

        返回:
            weight 数组
        """
        n = cov_matrix.shape[0]
        if n == 0:
            return np.array([])

        cov = cov_matrix.astype(float)
        cov += np.eye(n) * 1e-8  # 正定化

        target_risk = 1.0 / n  # 每个资产的目标风险贡献比例

        def objective(w):
            """最小化风险贡献方差"""
            portfolio_vol = np.sqrt(w @ cov @ w)
            if portfolio_vol < 1e-12:
                return 0.0

            # 各资产的边际风险贡献
            marginal_risk = cov @ w
            # 各资产的风险贡献
            risk_contrib = w * marginal_risk / portfolio_vol
            # 风险贡献占比
            risk_pct = risk_contrib / risk_contrib.sum() if risk_contrib.sum() > 0 else np.zeros(n)

            # 目标: 各风险贡献占比偏离均等的方差
            return np.sum((risk_pct - target_risk) ** 2)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        bounds = [(0.0, max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        weights = result.x
        weights[weights < 0.001] = 0.0
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights
