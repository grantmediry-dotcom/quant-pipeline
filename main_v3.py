"""
量化多因子策略 V3 — Agent 通信版入口
（保留 main.py 作为 V2 兼容入口）
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
