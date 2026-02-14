"""
单因子快速检验工具

用法：
    python quick_test.py <factor_name>
    python quick_test.py momentum_20d
    python quick_test.py --all           # 检验所有启用因子
    python quick_test.py --list          # 列出所有已注册因子

工作流程：
    1. 加载已缓存的日线行情（不重新拉取）
    2. 只计算指定的一个因子（~3秒）
    3. 计算 RankIC / IR / 分组收益（~5秒）
    4. 输出检验报告 + 通过/不通过判定

判定标准：
    |IC均值| >= 0.03 且 |IR| >= 0.5  → 通过
    否则 → 不通过（建议 disable）
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import DATA_DIR
from utils.helpers import load_parquet

# IC/IR 通过阈值
IC_THRESHOLD = 0.03
IR_THRESHOLD = 0.5


def load_daily_quotes() -> pd.DataFrame:
    """加载已缓存的清洗后日线行情"""
    path = os.path.join(DATA_DIR, "daily_quotes.parquet")
    df = load_parquet(path)
    if df is None:
        print("错误: 未找到日线行情缓存，请先运行 main_v3.py 拉取数据")
        sys.exit(1)
    return df


def compute_single_factor(df: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """只计算指定的一个因子"""
    import factors  # noqa: F401 — 触发所有因子注册
    from factor_framework.registry import FactorRegistry

    registry = FactorRegistry()

    # 检查因子是否存在
    try:
        instance = registry.create_instance(factor_name)
    except KeyError:
        all_factors = registry.list_factors()
        print(f"错误: 因子 '{factor_name}' 未注册")
        print(f"已注册因子: {all_factors}")
        sys.exit(1)

    meta = registry.get_metadata(factor_name)
    print(f"\n{'='*50}")
    print(f"  因子快检: {meta.display_name} ({factor_name})")
    print(f"  方向: {'值越大越好' if meta.direction == 1 else '值越小越好'}")
    print(f"  类别: {meta.category}")
    print(f"{'='*50}")

    work = df.copy()
    work = work.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 日收益率（因子计算基础）
    work["ret"] = work.groupby("ts_code")["close"].pct_change()

    # 计算因子
    ts = time.time()
    work[factor_name] = instance.compute(work)
    elapsed = time.time() - ts

    # 质量检验
    quality = instance.validate(work[factor_name])
    print(f"\n[计算] 耗时 {elapsed:.1f}s, NaN={quality['nan_ratio']:.1%}, Inf={quality['inf_count']}")

    # 清洗
    work[factor_name] = instance.clean(work[factor_name])

    # 排除 IPO 期
    if "is_ipo_period" in work.columns:
        n_before = len(work)
        work = work[work["is_ipo_period"] == 0]
        print(f"[清洗] 排除 IPO 期 {n_before - len(work)} 行")

    # 未来收益率
    work["fwd_ret_1m"] = work.groupby("ts_code")["close"].transform(
        lambda x: x.shift(-20) / x - 1
    )

    return work, instance, meta


def rank_ic_test(df: pd.DataFrame, factor_name: str) -> dict:
    """RankIC 检验"""
    clean = df.dropna(subset=[factor_name, "fwd_ret_1m"]).copy()

    clean["month"] = clean["trade_date"].str[:6]
    month_ends = clean.groupby("month")["trade_date"].max().values
    monthly = clean[clean["trade_date"].isin(month_ends)]

    ic_list = []
    for date, group in monthly.groupby("trade_date"):
        if len(group) < 30:
            continue
        corr, _ = stats.spearmanr(group[factor_name], group["fwd_ret_1m"])
        if not np.isnan(corr):
            ic_list.append({"trade_date": date, "ic": corr})

    if not ic_list:
        return {"ic_mean": 0, "ic_std": 0, "ir": 0, "ic_pos_pct": 0, "n_periods": 0}

    ic_series = pd.DataFrame(ic_list)["ic"]
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos_pct = (ic_series > 0).mean()

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "ic_pos_pct": ic_pos_pct,
        "n_periods": len(ic_list),
    }


def group_return_test(df: pd.DataFrame, factor_name: str, n_groups: int = 5) -> dict:
    """分组收益检验（分5组看单调性）"""
    clean = df.dropna(subset=[factor_name, "fwd_ret_1m"]).copy()

    clean["month"] = clean["trade_date"].str[:6]
    month_ends = clean.groupby("month")["trade_date"].max().values
    monthly = clean[clean["trade_date"].isin(month_ends)]

    group_rets = {g: [] for g in range(1, n_groups + 1)}

    for date, group in monthly.groupby("trade_date"):
        if len(group) < n_groups * 10:
            continue
        group = group.copy()
        group["group"] = pd.qcut(group[factor_name], n_groups, labels=False, duplicates="drop") + 1
        for g in range(1, n_groups + 1):
            g_data = group[group["group"] == g]
            if len(g_data) > 0:
                group_rets[g].append(g_data["fwd_ret_1m"].mean())

    result = {}
    for g in range(1, n_groups + 1):
        if group_rets[g]:
            result[f"G{g}"] = np.mean(group_rets[g])
        else:
            result[f"G{g}"] = 0

    # 多空收益 = G5 - G1
    result["long_short"] = result.get("G5", 0) - result.get("G1", 0)

    # 单调性检验（Spearman 相关）
    groups = list(range(1, n_groups + 1))
    means = [result[f"G{g}"] for g in groups]
    if len(means) >= 3:
        mono_corr, _ = stats.spearmanr(groups, means)
        result["monotonicity"] = mono_corr
    else:
        result["monotonicity"] = 0

    return result


def print_report(factor_name: str, meta, ic_result: dict, group_result: dict):
    """打印检验报告"""
    ic = ic_result["ic_mean"]
    ir = ic_result["ir"]

    passed = abs(ic) >= IC_THRESHOLD and abs(ir) >= IR_THRESHOLD

    print(f"\n{'─'*50}")
    print(f"  RankIC 检验 ({ic_result['n_periods']} 期)")
    print(f"{'─'*50}")
    print(f"  IC 均值:     {ic:+.4f}")
    print(f"  IC 标准差:   {ic_result['ic_std']:.4f}")
    print(f"  IR:          {ir:+.4f}")
    print(f"  IC>0 占比:   {ic_result['ic_pos_pct']:.1%}")

    print(f"\n{'─'*50}")
    print(f"  分组收益检验 (5组)")
    print(f"{'─'*50}")
    for g in range(1, 6):
        bar = "+" * int(abs(group_result.get(f"G{g}", 0)) * 200)
        sign = "+" if group_result.get(f"G{g}", 0) >= 0 else "-"
        print(f"  G{g}: {group_result.get(f'G{g}', 0):+.4f}  {sign}{bar}")
    print(f"  多空收益:    {group_result['long_short']:+.4f}")
    print(f"  单调性:      {group_result['monotonicity']:+.2f}")

    # 判定
    print(f"\n{'='*50}")
    if passed:
        print(f"  >>> 通过 <<<  |IC|={abs(ic):.4f}>={IC_THRESHOLD}  |IR|={abs(ir):.4f}>={IR_THRESHOLD}")
        print(f"  建议: 保留因子，方向={'正向' if (ic > 0) == (meta.direction > 0) else '需翻转(!)'}")
    else:
        reasons = []
        if abs(ic) < IC_THRESHOLD:
            reasons.append(f"|IC|={abs(ic):.4f} < {IC_THRESHOLD}")
        if abs(ir) < IR_THRESHOLD:
            reasons.append(f"|IR|={abs(ir):.4f} < {IR_THRESHOLD}")
        print(f"  >>> 不通过 <<<  {', '.join(reasons)}")
        print(f"  建议: 设为 enabled=False 或调整参数后重试")
    print(f"{'='*50}")

    return passed


def list_factors():
    """列出所有已注册因子"""
    import factors  # noqa: F401
    from factor_framework.registry import FactorRegistry
    registry = FactorRegistry()

    print(f"\n{'='*60}")
    print(f"  已注册因子列表")
    print(f"{'='*60}")

    all_names = registry.list_factors()
    for name in all_names:
        meta = registry.get_metadata(name)
        status = "ON " if meta.enabled else "OFF"
        ic_str = f"IC={meta.eval_ic_mean:+.4f}" if meta.eval_ic_mean is not None else "IC=未测"
        ir_str = f"IR={meta.eval_ir:+.4f}" if meta.eval_ir is not None else "IR=未测"
        print(f"  [{status}] {name:<20s} {meta.display_name:<10s} {ic_str}  {ir_str}")

    print(f"\n  共 {len(all_names)} 个因子，"
          f"启用 {len(registry.list_factors(enabled_only=True))} 个")


def test_all():
    """检验所有启用因子"""
    import factors  # noqa: F401
    from factor_framework.registry import FactorRegistry
    registry = FactorRegistry()

    enabled = registry.list_factors(enabled_only=True)
    df = load_daily_quotes()

    results = []
    for fname in enabled:
        work, instance, meta = compute_single_factor(df, fname)
        ic_result = rank_ic_test(work, fname)
        group_result = group_return_test(work, fname)
        passed = print_report(fname, meta, ic_result, group_result)
        results.append({"factor": fname, "passed": passed,
                        "ic": ic_result["ic_mean"], "ir": ic_result["ir"]})

    print(f"\n\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{tag}] {r['factor']:<20s} IC={r['ic']:+.4f}  IR={r['ir']:+.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python quick_test.py <factor_name>")
        print("      python quick_test.py --list")
        print("      python quick_test.py --all")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--list":
        list_factors()
    elif arg == "--all":
        df = load_daily_quotes()
        test_all()
    else:
        df = load_daily_quotes()
        work, instance, meta = compute_single_factor(df, arg)
        ic_result = rank_ic_test(work, arg)
        group_result = group_return_test(work, arg)
        print_report(arg, meta, ic_result, group_result)
