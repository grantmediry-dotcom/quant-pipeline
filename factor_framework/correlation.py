"""
因子相关性分析模块

自动检测因子间的截面 Rank 相关性，识别冗余因子对。
集成到 FactorAgent 的流程中，在单因子检验后自动运行。

输出：
- 因子相关性矩阵（保存 CSV + 热力图）
- 冗余因子对告警（|corr| > 阈值）
- 建议保留/去除的因子（基于 IR 择优）
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 相关性阈值：超过此值视为冗余对
CORR_THRESHOLD = 0.7


def compute_factor_corr_matrix(
    factors_df: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    """
    计算因子间的平均截面 Rank 相关系数矩阵

    方法：
    1. 对每个月末截面，计算所有因子对的 Spearman rank 相关系数
    2. 对所有截面取均值
    这比简单的全样本相关更稳健，避免时间序列自相关的干扰。
    """
    df = factors_df.copy()
    available = [f for f in factor_names if f in df.columns and df[f].notna().sum() > 0]

    if len(available) < 2:
        return pd.DataFrame()

    # 取月末截面
    df["month"] = df["trade_date"].str[:6]
    month_ends = df.groupby("month")["trade_date"].max().values
    monthly = df[df["trade_date"].isin(month_ends)]

    # 逐截面计算 rank 相关
    corr_accum = []
    for date, group in monthly.groupby("trade_date"):
        if len(group) < 100:
            continue
        # 截面 rank
        ranked = group[available].rank(pct=True)
        corr_mat = ranked.corr(method="spearman")
        corr_accum.append(corr_mat)

    if not corr_accum:
        return pd.DataFrame()

    # 取均值
    avg_corr = sum(corr_accum) / len(corr_accum)
    return avg_corr


def find_redundant_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> list[dict]:
    """
    找出相关性超过阈值的因子对

    返回：[{"factor_a": str, "factor_b": str, "corr": float}, ...]
    """
    pairs = []
    names = corr_matrix.columns.tolist()

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                pairs.append({
                    "factor_a": names[i],
                    "factor_b": names[j],
                    "corr": round(corr, 4),
                })

    return sorted(pairs, key=lambda x: -abs(x["corr"]))


def recommend_removals(
    redundant_pairs: list[dict],
    factor_ir: dict[str, float],
) -> list[str]:
    """
    对冗余因子对，推荐去除 IR 较低的那个

    参数：
        redundant_pairs: find_redundant_pairs 的输出
        factor_ir: {factor_name: IR}

    返回：建议去除的因子名列表（去重）
    """
    to_remove = set()
    kept = set()

    for pair in redundant_pairs:
        a, b = pair["factor_a"], pair["factor_b"]

        # 如果一方已被保留（在更高相关对中胜出），另一方直接去除
        if a in kept and b not in kept:
            to_remove.add(b)
            continue
        if b in kept and a not in kept:
            to_remove.add(a)
            continue

        # 比较 |IR|，保留更强的
        ir_a = abs(factor_ir.get(a, 0))
        ir_b = abs(factor_ir.get(b, 0))

        if ir_a >= ir_b:
            kept.add(a)
            to_remove.add(b)
        else:
            kept.add(b)
            to_remove.add(a)

    return sorted(to_remove)


def plot_corr_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: str,
    title: str = "Factor Rank Correlation Matrix",
):
    """保存相关性热力图"""
    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))

    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # 标签
    names = corr_matrix.columns.tolist()
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    # 数值标注
    for i in range(n):
        for j in range(n):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_correlation_analysis(
    factors_df: pd.DataFrame,
    factor_names: list[str],
    factor_ir: dict[str, float],
    output_dir: str,
    threshold: float = CORR_THRESHOLD,
) -> dict:
    """
    一键运行因子相关性分析（供 FactorAgent 调用）

    返回：
        {
            "corr_matrix": DataFrame,
            "redundant_pairs": [...],
            "recommended_removals": [...],
        }
    """
    print("[CorrelationAnalysis] 计算因子截面 Rank 相关性矩阵...")

    corr_matrix = compute_factor_corr_matrix(factors_df, factor_names)
    if corr_matrix.empty:
        print("[CorrelationAnalysis] 因子数不足，跳过相关性分析")
        return {"corr_matrix": pd.DataFrame(), "redundant_pairs": [], "recommended_removals": []}

    # 保存相关性矩阵
    corr_path = os.path.join(output_dir, "factor_correlation.csv")
    corr_matrix.to_csv(corr_path, encoding="utf-8-sig")

    # 保存热力图
    heatmap_path = os.path.join(output_dir, "factor_correlation_heatmap.png")
    plot_corr_heatmap(corr_matrix, heatmap_path)
    print(f"  相关性矩阵已保存: factor_correlation.csv")
    print(f"  热力图已保存: factor_correlation_heatmap.png")

    # 检测冗余对
    redundant = find_redundant_pairs(corr_matrix, threshold)
    if redundant:
        print(f"\n  [警告] 发现 {len(redundant)} 对冗余因子 (|corr| >= {threshold}):")
        for p in redundant:
            print(f"    {p['factor_a']} <-> {p['factor_b']}: corr={p['corr']}")
    else:
        print(f"  未发现冗余因子对 (阈值 |corr| >= {threshold})")

    # 推荐去除
    removals = recommend_removals(redundant, factor_ir)
    if removals:
        print(f"\n  [建议] 可考虑去除以下因子（冗余对中 IR 较弱者）:")
        for r in removals:
            ir_val = factor_ir.get(r, 0)
            print(f"    {r} (IR={ir_val:.4f})")

    return {
        "corr_matrix": corr_matrix,
        "redundant_pairs": redundant,
        "recommended_removals": removals,
    }
