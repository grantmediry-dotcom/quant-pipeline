"""
收益换手联动因子：收益率*价格 与 成交额的短期协方差

来源：华泰金工AI23，RankIC = 3.32%
逻辑：收益率放大（乘以价格作为dollar-return）与成交额同向波动
      → 追涨杀跌资金活跃 → 过热信号 → 未来收益更低
"""

import numpy as np
import pandas as pd
from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="ret_turn_cov_7d",
    display_name="收益换手联动",
    category="资金流",
    direction=1,        # compute已取反，值越大（追涨杀跌弱）→ 收益越高
    description="close*return 与 amount 的7日协方差，捕捉资金追涨杀跌行为",
    formula="-cov(close * daily_return, amount, 7)",
    parameters={"window": 7, "min_periods": 5},
    data_sources=["close", "pre_close", "amount"],
    lookback_days=7,
    author="quant_pipeline",
    status="research",
))
class RetTurnCov7D(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        window = self.metadata.parameters["window"]
        min_periods = self.metadata.parameters["min_periods"]

        ret = df.groupby("ts_code")["close"].pct_change()
        dollar_ret = df["close"] * ret

        result = pd.Series(np.nan, index=df.index)

        for code, group in df.groupby("ts_code"):
            idx = group.index
            dr = dollar_ret.loc[idx]
            amt = df.loc[idx, "amount"]
            cov = dr.rolling(window, min_periods=min_periods).cov(amt)
            result.loc[idx] = cov.values

        return -result
