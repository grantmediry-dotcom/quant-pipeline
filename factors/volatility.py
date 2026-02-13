"""波动率因子：过去20日收益率标准差（取反，低波动为正）"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="volatility_20d",
    display_name="20日波动率",
    category="风险",
    direction=1,        # 计算时已取反，值越高 = 波动越低 → 收益越高
    description="过去20日日收益率标准差取反，低波动异象：低波动股票往往有更高收益",
    formula="-daily_return.rolling(20).std()",
    parameters={"window": 20, "min_periods": 10},
    data_sources=["close"],
    lookback_days=21,
    author="quant_pipeline",
))
class Volatility20D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        # 计算日收益率
        ret = df.groupby("ts_code")["close"].pct_change()

        # 滚动标准差，取反（低波动 → 高值）
        vol = ret.groupby(df["ts_code"]).transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )
        return -vol
