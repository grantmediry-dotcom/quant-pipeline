"""
Alpha Agent - 因子合成与信号生成（注册表驱动版）

从 FactorRegistry 读取因子方向，自动完成方向调整后合成 Alpha 得分。
新增因子时无需修改此文件，只要注册到 registry 即可自动参与合成。
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 factors 包以触发注册
import factors  # noqa: F401
from factor_framework.registry import FactorRegistry


class AlphaAgent:
    """Alpha Agent：基于注册表的方向调整 + 等权合成"""

    def __init__(self):
        self.registry = FactorRegistry()
        self.enabled_factors = self.registry.get_enabled_factors()

        print("[AlphaAgent] 初始化完成")
        print(f"  启用因子: {list(self.enabled_factors.keys())}")
        for name, meta in self.enabled_factors.items():
            print(f"    {meta.display_name}({name}): direction={meta.direction:+d}")

    def composite_score(self, factor_scores: pd.DataFrame) -> pd.DataFrame:
        """
        方向调整后等权合成复合 Alpha 得分
        - 因子值乘以 metadata.direction（+1 或 -1）后取均值
        - 所有因子统一为"值越大 alpha 越高"
        """
        df = factor_scores.copy()
        available = [f for f in self.enabled_factors
                     if f in df.columns and df[f].notna().sum() > 0]

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

        df["alpha_score"] = df[adjusted_cols].mean(axis=1)
        df = df.drop(columns=adjusted_cols)

        return df[["ts_code", "alpha_score"]].dropna()
