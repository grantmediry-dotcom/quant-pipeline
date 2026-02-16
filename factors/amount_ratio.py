"""
成交额比率因子：近1月均成交额 / 近3月均成交额

来源：国盛证券
逻辑：近期成交额相对放大 → 交易过热/投机情绪上升 → 未来收益下降
"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="amount_ratio_1m3m",
    display_name="成交额比率",
    category="流动性",
    direction=-1,       # 近期成交额相对放大 → 过热 → 收益下降
    description="近22日均成交额与近66日均成交额的比值，捕捉流动性变化趋势",
    formula="mean(amount, 22) / mean(amount, 66)",
    parameters={"short_window": 22, "long_window": 66, "min_periods_long": 30},
    data_sources=["amount"],
    lookback_days=66,
    author="quant_pipeline",
    status="research",
))
class AmountRatio1M3M(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        short_w = self.metadata.parameters["short_window"]
        long_w = self.metadata.parameters["long_window"]
        min_p = self.metadata.parameters["min_periods_long"]

        amt_short = df.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(short_w, min_periods=short_w).mean()
        )
        amt_long = df.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(long_w, min_periods=min_p).mean()
        )

        return amt_short / amt_long.clip(lower=1)
