"""
因子基类模块（适配自 Ray 的 factor_framework/base.py）

核心改动：
- 简化 FactorMetadata，保留实用字段
- compute() 接收长表 DataFrame（ts_code, trade_date, ...），返回 Series
- validate() 做 Inf/NaN/极端值检验
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FactorMetadata:
    """因子元信息（精简版，保留 Ray 框架核心字段）"""

    # 必填
    name: str                           # 唯一标识，如 "momentum_20d"
    display_name: str                   # 中文展示名，如 "20日动量"
    category: str                       # 分类：动量/反转/流动性/风险/规模/...
    direction: int                      # +1=值越大越好，-1=值越小越好
    description: str                    # 因子描述

    # 计算相关
    formula: str = ""                   # 公式描述
    parameters: dict = field(default_factory=dict)   # 参数，如 {"lookback": 20}
    data_sources: list = field(default_factory=list)  # 所需字段，如 ["close", "vol"]
    lookback_days: int = 0              # 回溯天数

    # 生命周期
    author: str = ""
    version: str = "1.0.0"
    status: str = "research"            # research/validated/production/deprecated
    enabled: bool = True                # 是否参与合成

    # 时间戳
    created_at: str = ""
    updated_at: str = ""

    # 数据血缘（运行时填充）
    lineage: dict = field(default_factory=dict)

    # 评价指标缓存
    eval_ic_mean: Optional[float] = None
    eval_ir: Optional[float] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FactorMetadata":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class QuantitativeFactor(ABC):
    """
    量化因子抽象基类（适配自 Ray 的设计）

    与 Ray 的区别：
    - Ray 的 compute 返回宽表（index=日期, columns=股票），适合 RiceQuant
    - 我们的 compute 接收长表、返回 Series，适合 Tushare/AKShare 数据格式
    """

    def __init__(self, metadata: FactorMetadata):
        self.metadata = metadata

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算因子值。

        参数：
            df: 日线行情 DataFrame，含 ts_code, trade_date, open, high, low, close, vol, amount
                已按 (ts_code, trade_date) 排序
        返回：
            pd.Series，与 df 的 index 对齐，值为因子数值
        """

    def validate(self, values: pd.Series) -> dict:
        """
        质量检验（借鉴 Ray 的 validate 方法）
        """
        total = len(values)
        if total == 0:
            return {"valid": False, "nan_ratio": 1.0, "inf_count": 0, "total": 0}

        inf_count = int(np.isinf(values).sum())
        nan_count = int(values.isna().sum())
        nan_ratio = nan_count / total

        valid = nan_ratio <= 0.5 and inf_count == 0

        return {
            "valid": valid,
            "nan_ratio": nan_ratio,
            "inf_count": inf_count,
            "total": total,
        }

    def clean(self, values: pd.Series) -> pd.Series:
        """
        清洗因子值：替换 Inf → NaN，winsorize 到 1%/99% 分位
        """
        # 替换 Inf
        result = values.replace([np.inf, -np.inf], np.nan)

        # Winsorize
        valid = result.dropna()
        if len(valid) > 0:
            lower = valid.quantile(0.01)
            upper = valid.quantile(0.99)
            result = result.clip(lower=lower, upper=upper)

        return result
