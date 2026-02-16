"""
短长期均价比因子：5日VWAP / 60日VWAP

来源：东方证券（Rank IC: -0.054）、国盛证券
逻辑：短期均价高于长期均价 → 短期过热 → 均值回归 → 未来收益更低
"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="vwap_ratio_5d60d",
    display_name="均价比5/60",
    category="反转",
    direction=-1,       # 短期均价/长期均价 越高 → 过热 → 收益越低
    description="5日VWAP与60日VWAP的比值，捕捉短期相对长期的价格偏离程度",
    formula="(sum(amount,5)/sum(vol,5)) / (sum(amount,60)/sum(vol,60))",
    parameters={"short_window": 5, "long_window": 60, "min_periods_long": 30},
    data_sources=["amount", "vol"],
    lookback_days=60,
    author="quant_pipeline",
    status="research",
))
class VwapRatio5D60D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        short_w = self.metadata.parameters["short_window"]
        long_w = self.metadata.parameters["long_window"]
        min_p = self.metadata.parameters["min_periods_long"]

        # VWAP = sum(amount) / sum(vol)，按股票分组滚动
        amt_short = df.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(short_w, min_periods=short_w).sum()
        )
        vol_short = df.groupby("ts_code")["vol"].transform(
            lambda x: x.rolling(short_w, min_periods=short_w).sum()
        )
        amt_long = df.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(long_w, min_periods=min_p).sum()
        )
        vol_long = df.groupby("ts_code")["vol"].transform(
            lambda x: x.rolling(long_w, min_periods=min_p).sum()
        )

        vwap_short = amt_short / vol_short.clip(lower=1)
        vwap_long = amt_long / vol_long.clip(lower=1)

        return vwap_short / vwap_long.clip(lower=0.001)
