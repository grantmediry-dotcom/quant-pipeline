"""
Alpha Agent - factor synthesis and signal generation.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import factors  # noqa: F401
from factor_framework.registry import FactorRegistry
from core.agent_base import BaseAgent
from core.signal_bus import Signal
from utils.log import get_logger

logger = get_logger("agents.alpha_agent")


class AlphaAgent(BaseAgent):
    """Alpha Agent: registry-driven factor synthesis with dynamic weights."""

    agent_name = "AlphaAgent"

    def _setup_listeners(self):
        # P1: direct event-driven flow monitor -> alpha.
        self.listen("monitor.factor_weight_update", self._on_factor_weight_update)
        self.listen("monitor.factor_degraded", self._on_factor_degraded)

    def __init__(self):
        super().__init__()
        self.registry = FactorRegistry()
        self.enabled_factors = self.registry.get_enabled_factors()

        # None means equal-weight blend fallback.
        self.factor_weights = None
        self.excluded_factors = []

        logger.info(f"初始化完成，启用因子: {list(self.enabled_factors.keys())}")

    def _on_factor_weight_update(self, signal: Signal):
        payload = signal.payload or {}
        weights = payload.get("weights") or {}
        excluded = payload.get("excluded") or []
        if not isinstance(weights, dict):
            return
        self.set_factor_weights(weights, excluded)

    def _on_factor_degraded(self, signal: Signal):
        payload = signal.payload or {}
        degraded = payload.get("degraded_factors") or []
        if degraded:
            logger.info(f"收到退化因子: {degraded}")

    def set_factor_weights(self, weights: dict, excluded: list = None):
        """Receive factor weights from monitor feedback."""
        self.factor_weights = weights
        self.excluded_factors = excluded or []

        logger.info("收到 IC 权重更新:")
        for f, w in weights.items():
            logger.info(f"  {f}: {w:.1%}")
        if self.excluded_factors:
            logger.info(f"排除因子: {self.excluded_factors}")

    def composite_score(self, factor_scores: pd.DataFrame) -> pd.DataFrame:
        """Build composite alpha score using weighted or equal blend."""
        df = factor_scores.copy()

        available = [
            f
            for f in self.enabled_factors
            if f in df.columns and df[f].notna().sum() > 0 and f not in self.excluded_factors
        ]

        if not available:
            df["alpha_score"] = 0
            return df[["ts_code", "alpha_score"]]

        adjusted_cols = []
        for fname in available:
            col_name = f"_adj_{fname}"
            direction = self.enabled_factors[fname].direction
            df[col_name] = df[fname] * direction
            adjusted_cols.append(col_name)

        if self.factor_weights:
            df["alpha_score"] = 0.0
            for fname, col_name in zip(available, adjusted_cols):
                w = self.factor_weights.get(fname, 0.0)
                df["alpha_score"] += df[col_name] * w
        else:
            df["alpha_score"] = df[adjusted_cols].mean(axis=1)

        df = df.drop(columns=adjusted_cols)
        return df[["ts_code", "alpha_score"]].dropna()
