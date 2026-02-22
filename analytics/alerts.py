"""
告警模块

检测因子退化、回撤超限等异常，输出告警 JSON 文件。
"""

import json
import os
from datetime import datetime

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import ensure_dir
from utils.log import get_logger

logger = get_logger("analytics.alerts")


class AlertManager:
    """告警管理器"""

    def check_factor_degradation(
        self,
        factor_test: pd.DataFrame,
        ic_threshold: float = 0.03,
        ir_threshold: float = 0.3,
    ) -> list[dict]:
        """
        检测因子退化

        规则:
        - |IC| < ic_threshold: 因子预测力不足
        - |IR| < ir_threshold: 因子信号不稳定

        返回: [{factor_name, alert_type, value, threshold, message}, ...]
        """
        alerts = []

        # 列名自适应
        name_col = self._find_col(factor_test, ["因子名称", "factor_name", "name"])
        ic_col = self._find_col(factor_test, ["RankIC均值", "ic_mean", "ic"])
        ir_col = self._find_col(factor_test, ["IR", "ir"])

        if not all([name_col, ic_col, ir_col]):
            return alerts

        for _, row in factor_test.iterrows():
            fname = str(row[name_col])
            ic = abs(float(row[ic_col])) if pd.notna(row[ic_col]) else 0.0
            ir = abs(float(row[ir_col])) if pd.notna(row[ir_col]) else 0.0

            if ic < ic_threshold:
                alerts.append({
                    "factor_name": fname,
                    "alert_type": "low_ic",
                    "value": round(ic, 4),
                    "threshold": ic_threshold,
                    "message": f"{fname}: |IC|={ic:.4f} < {ic_threshold}",
                })

            if ir < ir_threshold:
                alerts.append({
                    "factor_name": fname,
                    "alert_type": "low_ir",
                    "value": round(ir, 4),
                    "threshold": ir_threshold,
                    "message": f"{fname}: |IR|={ir:.4f} < {ir_threshold}",
                })

        return alerts

    def check_drawdown(
        self,
        nav_df: pd.DataFrame,
        threshold: float = -0.15,
    ) -> dict | None:
        """
        检测回撤超限

        返回: alert dict 或 None
        """
        nav = nav_df["nav"].values
        peak = pd.Series(nav).cummax()
        drawdown = (pd.Series(nav) - peak) / peak
        max_dd = drawdown.min()

        if max_dd < threshold:
            return {
                "alert_type": "drawdown_exceeded",
                "value": round(float(max_dd), 4),
                "threshold": threshold,
                "message": f"最大回撤 {max_dd:.2%} 超过阈值 {threshold:.2%}",
            }
        return None

    def save_alerts(self, alerts: list[dict], output_dir: str):
        """保存告警到 JSON 文件"""
        if not alerts:
            return

        ensure_dir(output_dir)
        filepath = os.path.join(output_dir, "alerts.json")
        record = {
            "timestamp": datetime.now().isoformat(),
            "count": len(alerts),
            "alerts": alerts,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        logger.warning(f"发现 {len(alerts)} 条告警，已保存至 alerts.json")

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None
