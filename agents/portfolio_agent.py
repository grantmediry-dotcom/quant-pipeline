"""
Portfolio Agent - 组合构建
负责根据 Alpha 得分选股并分配权重
"""
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TOP_N
from core.agent_base import BaseAgent
from utils.log import get_logger

logger = get_logger("agents.portfolio_agent")


class PortfolioAgent(BaseAgent):
    """组合Agent：Top-N选股 + 等权配置"""

    agent_name = "PortfolioAgent"

    def __init__(self, top_n: int = TOP_N):
        super().__init__()
        self.top_n = top_n
        logger.info(f"初始化完成，每期选 {top_n} 只")

    def select(self, alpha_scores: pd.DataFrame) -> pd.DataFrame:
        """
        根据 alpha_score 选出 Top-N 股票，等权配置
        输入：含 ts_code, alpha_score 的 DataFrame
        输出：含 ts_code, weight 的 DataFrame
        """
        df = alpha_scores.sort_values("alpha_score", ascending=False).head(self.top_n)
        df = df.copy()
        df["weight"] = 1.0 / len(df)
        return df[["ts_code", "weight"]]
