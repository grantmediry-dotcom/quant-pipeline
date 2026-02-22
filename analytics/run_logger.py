"""
运行日志模块

将每次流水线运行的元信息追加到 JSONL 文件。
"""

import json
import os
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUT_DIR
from utils.helpers import ensure_dir


class RunLogger:
    """运行日志记录器"""

    def __init__(self, log_path: str = None):
        self.log_path = log_path or os.path.join(OUTPUT_DIR, "run_history.jsonl")
        ensure_dir(os.path.dirname(self.log_path))

    def log_run(
        self,
        run_id: str = None,
        mode: str = "full",
        duration: float = 0.0,
        status: str = "success",
        metrics: dict = None,
        error: str = None,
    ):
        """追加一条运行记录"""
        record = {
            "run_id": run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "duration_sec": round(duration, 1),
            "status": status,
            "metrics": metrics or {},
            "error": error,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_history(self, limit: int = 50) -> list[dict]:
        """读取最近 N 条运行记录"""
        if not os.path.exists(self.log_path):
            return []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return records[-limit:]
