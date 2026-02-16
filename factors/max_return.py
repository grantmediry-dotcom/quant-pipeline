"""
月内最大日涨幅因子（彩票效应）

来源：光大证券、国盛证券
逻辑：过去1个月内单日最大涨幅越高，说明投机/彩票偏好越强，未来收益越低
"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="max_return_1m",
    display_name="月内最大涨幅",
    category="动量",
    direction=-1,       # 最大日涨幅越高 → 彩票效应 → 未来收益越低
    description="过去22个交易日内单日最大涨幅，捕捉彩票效应/投机情绪",
    formula="max(daily_return, 22)",
    parameters={"window": 22, "min_periods": 10},
    data_sources=["close"],
    lookback_days=22,
    author="quant_pipeline",
    status="research",
))
class MaxReturn1M(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        ret = df.groupby("ts_code")["close"].pct_change()

        return ret.groupby(df["ts_code"]).transform(
            lambda x: x.rolling(window, min_periods=min_periods).max()
        )
