"""换手率因子：过去20日平均换手率（或成交额替代）"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="turnover_20d",
    display_name="20日换手率",
    category="流动性",
    direction=-1,       # 高换手 → 低收益
    description="过去20日平均换手率，衡量交易活跃度。无换手率数据时用成交额对数替代",
    formula="turnover_rate.rolling(20).mean() 或 log(amount).rolling(20).mean()",
    parameters={"window": 20, "min_periods": 10},
    data_sources=["turnover_rate", "amount"],
    lookback_days=20,
    author="quant_pipeline",
))
class Turnover20D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        if "turnover_rate" in df.columns and df["turnover_rate"].notna().sum() > 0:
            return df.groupby("ts_code")["turnover_rate"].transform(
                lambda x: x.rolling(window, min_periods=min_periods).mean()
            )
        else:
            return df.groupby("ts_code")["amount"].transform(
                lambda x: np.log(x.clip(lower=1)).rolling(window, min_periods=min_periods).mean()
            )
