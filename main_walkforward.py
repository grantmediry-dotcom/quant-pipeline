"""
Walk-Forward 样本外验证 — 入口文件

用法：
    python main_walkforward.py              # 默认参数
    python main_walkforward.py --fast       # 快速模式（1年训练，3个月测试）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.walk_forward import WalkForwardValidator, generate_windows


def main():
    if "--fast" in sys.argv:
        # 快速模式：缩短窗口，用于调试
        windows = generate_windows(train_months=12, test_months=3, step_months=3)
        print("[WalkForward] 快速模式：1年训练 + 3个月测试")
    else:
        windows = generate_windows()

    validator = WalkForwardValidator(windows=windows)
    validator.run()


if __name__ == "__main__":
    main()
