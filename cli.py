"""
量化多因子策略 CLI

用法:
    python cli.py run --mode full          # 全量回测
    python cli.py run --mode walkforward   # Walk-Forward 验证
    python cli.py run --mode factors-only  # 仅计算因子
    python cli.py sync                     # 仅拉取数据
    python cli.py test                     # 运行测试
"""

import argparse
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.log import get_logger

logger = get_logger("cli")


def cmd_run(args):
    """执行策略流水线"""
    mode = args.mode
    start = datetime.now()
    logger.info(f"启动流水线，模式: {mode}")

    try:
        if mode == "full":
            from core.orchestrator import Orchestrator
            Orchestrator().run()

        elif mode == "walkforward":
            from core.walk_forward import WalkForwardValidator, generate_windows
            if args.fast:
                windows = generate_windows(train_months=12, test_months=3, step_months=3)
                logger.info("快速模式：1年训练 + 3个月测试")
            else:
                windows = generate_windows()
            WalkForwardValidator(windows=windows).run()

        elif mode == "factors-only":
            from agents.data_agent import DataAgent
            from agents.factor_agent import FactorAgent

            data_agent = DataAgent()
            data_agent.update_all()
            daily_quotes = data_agent.get_daily_quotes()

            factor_agent = FactorAgent(daily_quotes)
            factor_agent.compute_factors()
            factor_agent.single_factor_test()
            logger.info("因子计算 + 检验完成")

        else:
            logger.error(f"未知模式: {mode}")
            return

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"流水线完成，耗时 {elapsed:.1f}s")

        # 记录运行日志
        _log_run(mode, elapsed, "success")

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        logger.error(f"流水线失败: {e}")
        _log_run(mode, elapsed, "failed", str(e))
        raise


def cmd_sync(args):
    """仅拉取数据"""
    from agents.data_agent import DataAgent
    agent = DataAgent()
    agent.update_all()
    logger.info("数据同步完成")


def cmd_test(args):
    """运行 pytest"""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
    )
    sys.exit(result.returncode)


def _log_run(mode: str, duration: float, status: str, error: str = None):
    """追加运行记录到 run_history.jsonl"""
    try:
        from analytics.run_logger import RunLogger
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        RunLogger().log_run(run_id=run_id, mode=mode, duration=duration, status=status, error=error)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="量化多因子策略 CLI")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # run
    run_parser = subparsers.add_parser("run", help="执行策略流水线")
    run_parser.add_argument(
        "--mode", choices=["full", "walkforward", "factors-only"],
        default="full", help="运行模式",
    )
    run_parser.add_argument("--fast", action="store_true", help="快速模式（仅 walkforward）")
    run_parser.set_defaults(func=cmd_run)

    # sync
    sync_parser = subparsers.add_parser("sync", help="仅拉取数据")
    sync_parser.set_defaults(func=cmd_sync)

    # test
    test_parser = subparsers.add_parser("test", help="运行测试")
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
