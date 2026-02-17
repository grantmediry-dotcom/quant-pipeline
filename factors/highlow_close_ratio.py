"""
多空力量比因子：(最高价+最低价)/收盘价 的短期累积

来源：华泰金工AI21，RankIC = 2.45%
逻辑：(H+L)/C > 2 说明日内多空拉锯后收盘偏低（空头占优）
      累积值越高 → 持续收盘偏低 → 存在超跌反弹空间（方向需验证）
"""

import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="highlow_close_sum_5d",
    display_name="多空力量比",
    category="技术",
    direction=-1,       # 值越大 → 持续收盘偏低 → 研报方向为负
    enabled=False,      # IR=0.24 未通过检验
    description="过去5日 (high+low)/close 累加值，衡量日内多空力量对比的持续性",
    formula="ts_sum((high + low) / close, 5)",
    parameters={"window": 5, "min_periods": 5},
    data_sources=["high", "low", "close"],
    lookback_days=5,
    author="quant_pipeline",
    status="research",
))
class HighLowCloseSum5D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        daily_ratio = (df["high"] + df["low"]) / df["close"].clip(lower=0.01)

        return daily_ratio.groupby(df["ts_code"]).transform(
            lambda x: x.rolling(window, min_periods=min_periods).sum()
        )
