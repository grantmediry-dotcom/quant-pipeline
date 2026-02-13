"""市值因子：总市值对数取反（小市值为正）"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="size_factor",
    display_name="市值因子",
    category="规模",
    direction=1,        # 计算时已取反，值越高 = 市值越小 → 收益越高（小市值效应）
    description="总市值对数取反，无市值数据时用60日平均成交额对数替代",
    formula="-log(total_mv) 或 -log(amount).rolling(60).mean()",
    parameters={"window": 60, "min_periods": 20},
    data_sources=["total_mv", "amount"],
    lookback_days=60,
    author="quant_pipeline",
))
class SizeFactor(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "total_mv" in df.columns and df["total_mv"].notna().sum() > 0:
            return -np.log(df["total_mv"].clip(lower=1))
        else:
            window = self.metadata.parameters["window"]
            min_periods = self.metadata.parameters["min_periods"]
            return -df.groupby("ts_code")["amount"].transform(
                lambda x: np.log(x.clip(lower=1)).rolling(window, min_periods=min_periods).mean()
            )
