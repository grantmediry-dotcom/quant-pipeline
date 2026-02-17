"""
短期成交量波动率因子

来源：华泰金工AI21（遗传规划因子挖掘）
逻辑：成交量波动大的股票，交易不稳定，投机性强，未来收益较低
与 volatility_20d 的区别：本因子衡量量的波动，而非价的波动
"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="volume_std_5d",
    display_name="短期量波动",
    category="流动性",
    direction=-1,       # 量波动越大 → 投机性越强 → 收益越低
    description="过去5日成交量标准差，衡量交易活跃度的稳定性",
    formula="-ts_stddev(volume, 5)",
    parameters={"window": 5, "min_periods": 4},
    data_sources=["vol"],
    lookback_days=5,
    author="quant_pipeline",
    status="research",
))
class VolumeStd5D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        return -df.groupby("ts_code")["vol"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )
