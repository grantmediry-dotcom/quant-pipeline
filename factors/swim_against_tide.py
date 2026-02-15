"""
水中行舟因子（日频近似版）

原始因子（方正证券）：
    使用分钟级数据，计算个股分钟成交量与市场分钟成交量的 Pearson 相关系数。
    低相关 = 个股交易节奏独立于市场 = 可能有知情交易者驱动 = 看涨信号。

日频近似：
    用日度成交额变化率（而非水平）计算个股与市场的滚动相关系数。
    变化率比水平值更能捕捉"节奏"信息——level相关往往被规模效应主导，
    而 pct_change 的相关更接近原始因子衡量的"交易节奏同步性"。

方向：-1（相关性越低越好）
"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="swim_against_tide",
    display_name="水中行舟",
    category="资金流",
    direction=1,           # 日频下高相关 = 好（与分钟级逻辑相反）
    description="个股日成交额变化率与市场总成交额变化率的滚动相关系数。"
                "日频下高相关 = 市场关注度高/流动性好 = 看涨信号",
    formula="corr(pct_change(stock_amount), pct_change(market_amount), window=20)",
    parameters={"window": 20, "min_periods": 10},
    data_sources=["amount"],
    lookback_days=21,
    author="quant_pipeline",
    status="research",
))
class SwimAgainstTide(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        # 全市场每日总成交额 → 日度变化率
        daily_market = df.groupby("trade_date")["amount"].sum()
        market_chg = daily_market.pct_change()
        # 将市场变化率映射回每行
        market_chg_mapped = df["trade_date"].map(market_chg)

        # 个股成交额日度变化率（组内 pct_change）
        stock_chg = df.groupby("ts_code")["amount"].pct_change()

        # 逐股计算滚动相关系数
        result = pd.Series(np.nan, index=df.index)

        for code, group in df.groupby("ts_code"):
            idx = group.index
            s = stock_chg.loc[idx]
            m = market_chg_mapped.loc[idx]
            corr = s.rolling(window, min_periods=min_periods).corr(m)
            result.loc[idx] = corr

        return result
