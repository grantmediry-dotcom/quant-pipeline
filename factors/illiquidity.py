"""
Amihud非流动性因子：日均 |收益率| / 成交额

来源：多篇研报（东方证券、光大证券、国盛证券）
逻辑：流动性差的股票有流动性溢价，未来收益更高
"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="illiquidity_20d",
    display_name="非流动性",
    category="流动性",
    direction=1,        # 非流动性越高 → 流动性溢价 → 收益越高
    description="Amihud非流动性：过去20日 mean(|daily_return| / amount)，衡量价格冲击成本",
    formula="mean(|close/pre_close - 1| / amount, 20)",
    parameters={"window": 20, "min_periods": 10},
    data_sources=["close", "pre_close", "amount"],
    lookback_days=20,
    author="quant_pipeline",
    status="research",
))
class Illiquidity20D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        # |日收益率| / 成交额（成交额单位：千元）
        ret = df.groupby("ts_code")["close"].pct_change()
        ratio = ret.abs() / df["amount"].clip(lower=1)

        return ratio.groupby(df["ts_code"]).transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
