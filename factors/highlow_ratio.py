"""
最高最低价比因子：过去N日最高价/最低价

来源：光大证券、华泰证券
逻辑：价格振幅越大 → 不确定性越高 → 未来收益越低（低波动异象的变体）
"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="highlow_ratio_20d",
    display_name="高低价比",
    category="风险",
    direction=-1,       # 振幅越大 → 收益越低
    description="过去20日最高价与最低价的比值，衡量价格区间振幅",
    formula="max(high, 20) / min(low, 20)",
    parameters={"window": 20, "min_periods": 10},
    data_sources=["high", "low"],
    lookback_days=20,
    author="quant_pipeline",
    status="research",
))
class HighLowRatio20D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        rolling_high = df.groupby("ts_code")["high"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).max()
        )
        rolling_low = df.groupby("ts_code")["low"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).min()
        )

        return rolling_high / rolling_low.clip(lower=0.01)
