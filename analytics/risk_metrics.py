"""
风险分析模块

提供滚动风险指标、VaR/CVaR、尾部风险统计、Sortino/Calmar 比率。
"""

import numpy as np
import pandas as pd
from scipy import stats


class RiskAnalyzer:
    """风险分析器"""

    def rolling_metrics(
        self,
        nav_df: pd.DataFrame,
        windows: list[int] = None,
    ) -> pd.DataFrame:
        """
        计算滚动风险指标

        参数:
            nav_df: 含 trade_date, nav 的 DataFrame
            windows: 滚动窗口列表（交易日数），默认 [30, 60, 90]

        返回:
            DataFrame 含 trade_date, vol_{w}d, sharpe_{w}d 列
        """
        if windows is None:
            windows = [30, 60, 90]

        df = nav_df[["trade_date", "nav"]].copy()
        df["daily_ret"] = df["nav"].pct_change()

        for w in windows:
            roll = df["daily_ret"].rolling(w, min_periods=max(w // 2, 10))
            df[f"vol_{w}d"] = roll.std() * np.sqrt(252)
            roll_mean = roll.mean() * 252
            roll_std = roll.std() * np.sqrt(252)
            df[f"sharpe_{w}d"] = np.where(
                roll_std > 0, (roll_mean - 0.02) / roll_std, 0
            )

        return df.drop(columns=["daily_ret"])

    def var_parametric(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        参数法 VaR（假设正态分布）

        返回: 负值（表示损失），如 -0.025 表示 95% 置信下单日最大损失 2.5%
        """
        mu = returns.mean()
        sigma = returns.std()
        if sigma == 0:
            return 0.0
        z = stats.norm.ppf(1 - confidence)
        return float(mu + z * sigma)

    def var_historical(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        历史法 VaR

        返回: 负值（表示损失）
        """
        if len(returns) == 0:
            return 0.0
        return float(returns.quantile(1 - confidence))

    def cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        条件 VaR (CVaR / Expected Shortfall)

        超过 VaR 阈值后的平均损失
        """
        var = self.var_historical(returns, confidence)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return var
        return float(tail.mean())

    def tail_risk(self, returns: pd.Series) -> dict:
        """
        尾部风险统计

        返回: {skewness, kurtosis, downside_std}
        """
        skew = float(returns.skew()) if len(returns) > 2 else 0.0
        kurt = float(returns.kurtosis()) if len(returns) > 3 else 0.0
        downside = self._downside_deviation(returns)
        return {
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "downside_std": round(downside, 6),
        }

    def sortino_ratio(self, returns: pd.Series, rf_daily: float = 0.02 / 252) -> float:
        """
        Sortino 比率 = (年化收益 - 无风险) / 年化下行标准差
        """
        ann_ret = returns.mean() * 252
        downside_std = self._downside_deviation(returns) * np.sqrt(252)
        if downside_std == 0:
            return 0.0
        return float((ann_ret - 0.02) / downside_std)

    def calmar_ratio(self, nav_df: pd.DataFrame) -> float:
        """
        Calmar 比率 = 年化收益 / |最大回撤|
        """
        nav = nav_df["nav"].values
        n_days = len(nav)
        if n_days < 2:
            return 0.0

        total_ret = nav[-1] / nav[0] - 1
        ann_ret = (1 + total_ret) ** (252 / n_days) - 1

        peak = pd.Series(nav).cummax()
        drawdown = (pd.Series(nav) - peak) / peak
        max_dd = abs(drawdown.min())

        if max_dd == 0:
            return 0.0
        return float(ann_ret / max_dd)

    def full_report(self, nav_df: pd.DataFrame) -> dict:
        """
        综合风险报告

        返回: 所有风险指标的 dict（值为格式化字符串）
        """
        daily_ret = nav_df["nav"].pct_change().dropna()

        var_p = self.var_parametric(daily_ret)
        var_h = self.var_historical(daily_ret)
        cvar_val = self.cvar(daily_ret)
        tail = self.tail_risk(daily_ret)
        sortino = self.sortino_ratio(daily_ret)
        calmar = self.calmar_ratio(nav_df)

        return {
            "Sortino比率": f"{sortino:.2f}",
            "Calmar比率": f"{calmar:.2f}",
            "VaR(95%)参数法": f"{var_p:.4f}",
            "VaR(95%)历史法": f"{var_h:.4f}",
            "CVaR(95%)": f"{cvar_val:.4f}",
            "偏度": f"{tail['skewness']:.4f}",
            "峰度": f"{tail['kurtosis']:.4f}",
            "下行波动率(年化)": f"{tail['downside_std'] * np.sqrt(252):.2%}",
        }

    @staticmethod
    def _downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
        """
        下行标准差（Minimum Acceptable Return = mar）
        只统计低于 mar 的收益的标准差
        """
        downside = returns[returns < mar] - mar
        if len(downside) == 0:
            return 0.0
        return float(np.sqrt((downside ** 2).mean()))
