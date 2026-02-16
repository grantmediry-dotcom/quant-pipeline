"""
价量相关性因子：收盘价与成交量的滚动相关系数

来源：国盛证券
逻辑：价格与成交量高度正相关 → 量价齐升趋势末期 → 趋势反转风险 → 未来收益更低
"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="price_volume_corr_20d",
    display_name="价量相关性",
    category="资金流",
    direction=-1,       # 价量正相关越强 → 趋势末期 → 收益越低
    description="过去20日收盘价与成交量的Pearson相关系数，捕捉量价背离/同步信号",
    formula="corr(close, vol, 20)",
    parameters={"window": 20, "min_periods": 10},
    data_sources=["close", "vol"],
    lookback_days=20,
    author="quant_pipeline",
    status="research",
))
class PriceVolumeCorr20D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        result = pd.Series(np.nan, index=df.index)

        for code, group in df.groupby("ts_code"):
            idx = group.index
            corr = group["close"].rolling(window, min_periods=min_periods).corr(
                group["vol"]
            )
            result.loc[idx] = corr.values

        return result
