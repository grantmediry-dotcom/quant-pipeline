"""
Alpha Agent - 因子合成与信号生成（V3：支持 IC 加权）

V3 新增：set_factor_weights() — 接收 MonitorAgent 的权重调整信号
composite_score() 支持 IC 加权合成（有权重时）或等权（兼容 V2）
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import factors  # noqa: F401  触发注册
from factor_framework.registry import FactorRegistry
from core.agent_base import BaseAgent


class AlphaAgent(BaseAgent):
    """Alpha Agent：注册表驱动 + 支持 IC 加权合成"""

    agent_name = "AlphaAgent"

    def __init__(self):
        super().__init__()
        self.registry = FactorRegistry()
        self.enabled_factors = self.registry.get_enabled_factors()

        # V3: 自定义权重（默认 None = 等权）
        self.factor_weights = None
        self.excluded_factors = []

        print("[AlphaAgent] 初始化完成")
        print(f"  启用因子: {list(self.enabled_factors.keys())}")

    def set_factor_weights(self, weights: dict, excluded: list = None):
        """
        V3: 接收 MonitorAgent 的权重调整建议
        weights: {factor_name: float} — 权重字典，总和为1
        excluded: [factor_name, ...] — 被排除的因子
        """
        self.factor_weights = weights
        self.excluded_factors = excluded or []

        print(f"  [AlphaAgent] 接受IC加权调整:")
        for f, w in weights.items():
            print(f"    {f}: {w:.1%}")
        if self.excluded_factors:
            print(f"  [AlphaAgent] 排除因子: {self.excluded_factors}")

    def composite_score(self, factor_scores: pd.DataFrame) -> pd.DataFrame:
        """
        方向调整后合成 Alpha 得分
        V3: 有 factor_weights → IC 加权；无 → 等权（兼容 V2）
        """
        df = factor_scores.copy()

        # 确定可用因子（排除衰减因子）
        available = [f for f in self.enabled_factors
                     if f in df.columns
                     and df[f].notna().sum() > 0
                     and f not in self.excluded_factors]

        if not available:
            df["alpha_score"] = 0
            return df[["ts_code", "alpha_score"]]

        # 方向调整
        adjusted_cols = []
        for fname in available:
            col_name = f"_adj_{fname}"
            direction = self.enabled_factors[fname].direction
            df[col_name] = df[fname] * direction
            adjusted_cols.append(col_name)

        # IC 加权 or 等权
        if self.factor_weights:
            df["alpha_score"] = 0.0
            for fname, col_name in zip(available, adjusted_cols):
                w = self.factor_weights.get(fname, 0.0)
                df["alpha_score"] += df[col_name] * w
        else:
            df["alpha_score"] = df[adjusted_cols].mean(axis=1)

        df = df.drop(columns=adjusted_cols)
        return df[["ts_code", "alpha_score"]].dropna()
