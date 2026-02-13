"""
因子开发模板（适配自 Ray 的 factors/_template.py）

使用方法：
    1. 复制此文件到 factors/ 下并重命名（如 my_factor.py）
    2. 修改 FactorMetadata 中的元信息
    3. 在 compute() 中实现因子计算逻辑
    4. 在 factors/__init__.py 中添加 import
    5. 运行 main.py，新因子自动参与计算和回测
"""

import pandas as pd

from factor_framework.base import FactorMetadata, QuantitativeFactor
from factor_framework.registry import register_factor


@register_factor(FactorMetadata(
    name="my_factor_20d",
    display_name="我的因子(20日)",
    category="自定义",
    direction=1,            # +1=值越大越好，-1=值越小越好
    description="因子描述：简要说明因子含义和逻辑",
    formula="close_t / close_{t-20} - 1",
    parameters={"lookback": 20},
    data_sources=["close"],
    lookback_days=20,
    author="your_name",
    enabled=False,          # 模板默认不启用
))
class MyFactor(QuantitativeFactor):

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算因子值。

        参数 df 包含列：ts_code, trade_date, open, high, low, close, vol, amount
        已按 (ts_code, trade_date) 排序。

        返回 pd.Series，index 与 df.index 对齐。
        """
        lookback = self.metadata.parameters["lookback"]
        return df.groupby("ts_code")["close"].transform(
            lambda x: x / x.shift(lookback) - 1
        )
