"""
均价趋势量价联动因子：VWAP累积 × 成交量-VWAP协方差

来源：华泰金工AI23，RankIC = 4.03%（本轮研报中最高IC的GP因子之一）
逻辑：VWAP上升趋势中，成交量与VWAP正协方差 → 放量推价 → 过热 → 未来下跌
"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="vwap_vol_cov_5d",
    display_name="均价量价联动",
    category="资金流",
    direction=1,        # compute已取反，值越大（放量推价弱）→ 收益越高
    description="5日VWAP累计 × 3日成交量与VWAP协方差的乘积，衡量放量推价强度",
    formula="-(ts_sum(vwap, 5) * cov(volume, vwap, 3))",
    parameters={"sum_window": 5, "cov_window": 3, "min_periods": 3},
    data_sources=["amount", "vol"],
    lookback_days=5,
    author="quant_pipeline",
    status="research",
))
class VwapVolCov5D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        sum_w = self.metadata.parameters["sum_window"]
        cov_w = self.metadata.parameters["cov_window"]
        min_p = self.metadata.parameters["min_periods"]

        vwap = df["amount"] / df["vol"].clip(lower=1)

        result = pd.Series(np.nan, index=df.index)

        for code, group in df.groupby("ts_code"):
            idx = group.index
            v = vwap.loc[idx]
            vol = df.loc[idx, "vol"]

            vwap_sum = v.rolling(sum_w, min_periods=min_p).sum()
            vol_vwap_cov = vol.rolling(cov_w, min_periods=min_p).cov(v)

            result.loc[idx] = -(vwap_sum * vol_vwap_cov).values

        return result
