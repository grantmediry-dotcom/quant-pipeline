"""
开盘价/成交量排名比因子

来源：华泰金工AI23（遗传规划因子挖掘），年化多空超额 7.57%
逻辑：开盘价排名高但成交量排名低的股票 = "贵而安静" → 筹码集中、关注度低 → 未来收益更高
这是纯截面因子，每日横截面计算
"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="open_vol_rank_div",
    display_name="开盘量排名比",
    category="资金流",
    direction=1,        # 比值越高（贵而安静）→ 收益越高
    enabled=False,      # IR=0.17 未通过检验
    description="开盘价截面排名 / 成交量截面排名，捕捉'贵而安静'的筹码集中特征",
    formula="rank(open) / rank(volume)",
    parameters={},
    data_sources=["open", "vol"],
    lookback_days=0,
    author="quant_pipeline",
    status="research",
))
class OpenVolRankDiv(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # 截面排名：每日在所有股票中排百分位
        open_rank = df.groupby("trade_date")["open"].rank(pct=True)
        vol_rank = df.groupby("trade_date")["vol"].rank(pct=True)

        # 避免除零
        return open_rank / vol_rank.clip(lower=0.01)
