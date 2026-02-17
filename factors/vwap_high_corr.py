"""
VWAP/高价比与高价相关性因子

来源：华泰金工AI21（遗传规划因子挖掘），RankIC = 3.83%, IC_IR = 0.94
逻辑：VWAP接近最高价说明大部分成交发生在高位，
      若这个比值与最高价正相关，说明高价时成交密集 → 机构积极买入 → 未来收益更高
"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="vwap_high_corr_10d",
    display_name="VWAP高价比相关",
    category="资金流",
    direction=1,        # 相关性越高 → 机构高位买入信号 → 收益越高
    description="VWAP/最高价比值与最高价的10日滚动相关系数，捕捉成交重心偏高的持续性",
    formula="corr(vwap/high, high, 10)",
    parameters={"window": 10, "min_periods": 7},
    data_sources=["amount", "vol", "high"],
    lookback_days=10,
    author="quant_pipeline",
    status="research",
))
class VwapHighCorr10D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        vwap = df["amount"] / df["vol"].clip(lower=1)
        ratio = vwap / df["high"].clip(lower=0.01)

        result = pd.Series(np.nan, index=df.index)

        for code, group in df.groupby("ts_code"):
            idx = group.index
            r = ratio.loc[idx]
            h = df.loc[idx, "high"]
            corr = r.rolling(window, min_periods=min_periods).corr(h)
            result.loc[idx] = corr.values

        return result
