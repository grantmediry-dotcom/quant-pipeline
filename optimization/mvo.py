"""
均值-方差优化 (MVO)

目标函数: min  -w'α + λ * w'Σw
约束: Σw = 1, 0 ≤ w_i ≤ max_weight
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MeanVarianceOptimizer:
    """均值-方差组合优化器"""

    def optimize(
        self,
        alpha_scores: pd.Series,
        cov_matrix: np.ndarray,
        max_weight: float = 0.10,
        risk_aversion: float = 1.0,
    ) -> pd.Series:
        """
        均值-方差优化

        参数:
            alpha_scores: Series(index=ts_code, values=alpha_score)
            cov_matrix: n×n 协方差矩阵 (numpy array)
            max_weight: 单只股票最大权重
            risk_aversion: 风险厌恶系数（越大越保守）

        返回:
            Series(index=ts_code, values=weight)
        """
        n = len(alpha_scores)
        if n == 0:
            return pd.Series(dtype=float)

        alpha = alpha_scores.values.astype(float)

        # 确保协方差矩阵正定（加微小对角项）
        cov = cov_matrix.astype(float)
        cov += np.eye(n) * 1e-8

        # 目标: min -w'α + λ * w'Σw
        def objective(w):
            return -w @ alpha + risk_aversion * w @ cov @ w

        # 梯度
        def gradient(w):
            return -alpha + 2 * risk_aversion * cov @ w

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Σw = 1
        ]

        # 边界: 0 ≤ w_i ≤ max_weight
        bounds = [(0.0, max_weight)] * n

        # 初始值: 等权
        w0 = np.ones(n) / n

        result = minimize(
            objective, w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        weights = result.x
        # 清理微小权重（< 0.1%）
        weights[weights < 0.001] = 0.0
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return pd.Series(weights, index=alpha_scores.index)

    @staticmethod
    def estimate_covariance(
        daily_quotes: pd.DataFrame,
        codes: list[str],
        lookback: int = 60,
    ) -> np.ndarray:
        """
        估算协方差矩阵（基于历史日收益率）

        使用 Ledoit-Wolf 收缩估计提高稳定性:
        Σ_shrunk = (1-δ) * Σ_sample + δ * F
        其中 F = diag(σ²) 是对角目标矩阵
        """
        # 获取最近 lookback 天的收益率
        recent_dates = sorted(daily_quotes["trade_date"].unique())[-lookback:]
        subset = daily_quotes[
            (daily_quotes["ts_code"].isin(codes))
            & (daily_quotes["trade_date"].isin(recent_dates))
        ]

        # 构建收益率矩阵 (日期 × 股票)
        pivot = subset.pivot_table(index="trade_date", columns="ts_code", values="close")
        returns = pivot.pct_change().dropna()

        # 只保留有数据的股票
        returns = returns.reindex(columns=codes)
        returns = returns.fillna(0)

        if len(returns) < 10 or len(returns.columns) < 2:
            # 数据不足，返回单位矩阵
            return np.eye(len(codes)) * 0.04 / 252  # 假设 20% 年化波动率

        # Ledoit-Wolf 收缩
        sample_cov = returns.cov().values
        n = sample_cov.shape[0]

        # 目标矩阵: 对角
        target = np.diag(np.diag(sample_cov))

        # 收缩系数（简化估计）
        delta = min(0.5, 2.0 / n)

        shrunk_cov = (1 - delta) * sample_cov + delta * target
        return shrunk_cov
