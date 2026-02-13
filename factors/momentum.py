"""动量因子：过去20日收益率"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="momentum_20d",
    display_name="20日动量",
    category="动量",
    direction=-1,       # A股短期反转效应：高动量 → 低未来收益
    description="过去20个交易日的价格涨跌幅，衡量中短期趋势强度",
    formula="close_t / close_{t-20} - 1",
    parameters={"lookback": 20},
    data_sources=["close"],
    lookback_days=20,
    author="quant_pipeline",
))
class Momentum20D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        lookback = self.metadata.parameters["lookback"]
        return df.groupby("ts_code")["close"].transform(
            lambda x: x / x.shift(lookback) - 1
        )
