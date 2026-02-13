"""反转因子：过去5日收益率取反"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="reversal_5d",
    display_name="5日反转",
    category="反转",
    direction=1,        # 计算时已取反，值越高 = 过去跌越多 → 未来涨
    description="过去5个交易日收益率取反，捕捉短期均值回归效应",
    formula="-(close_t / close_{t-5} - 1)",
    parameters={"lookback": 5},
    data_sources=["close"],
    lookback_days=5,
    author="quant_pipeline",
))
class Reversal5D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        lookback = self.metadata.parameters["lookback"]
        return -df.groupby("ts_code")["close"].transform(
            lambda x: x / x.shift(lookback) - 1
        )
