"""公用工具函数"""
import os
import shutil
import pandas as pd


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def save_parquet(df: pd.DataFrame, filepath: str):
    """
    保存 DataFrame 为 parquet（原子写入版，借鉴 Ray 的 FactorStore）

    流程：写入 .tmp 临时文件 → rename 为正式文件
    防止写入中断导致文件损坏
    """
    ensure_dir(os.path.dirname(filepath))
    tmp_path = filepath + ".tmp"
    df.to_parquet(tmp_path, index=False)
    _atomic_move(tmp_path, filepath)


def _atomic_move(src: str, dst: str):
    """原子移动文件（Windows 兼容版，参考 Ray 的 _atomic_write）"""
    try:
        os.replace(src, dst)
    except OSError:
        # Windows 兜底：先备份、再移动、最后清理
        bak = dst + ".bak"
        try:
            if os.path.exists(dst):
                os.rename(dst, bak)
            shutil.move(src, dst)
            if os.path.exists(bak):
                os.unlink(bak)
        except Exception:
            if not os.path.exists(dst) and os.path.exists(bak):
                os.rename(bak, dst)
            raise


def load_parquet(filepath: str) -> pd.DataFrame | None:
    """加载 parquet 文件，不存在返回 None"""
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    return None


def date_range_monthly(start: str, end: str) -> list[str]:
    """生成月末交易日列表（YYYYMMDD格式）"""
    dates = pd.date_range(start, end, freq="ME")
    return [d.strftime("%Y%m%d") for d in dates]


def get_month_end_dates(trade_dates) -> list[str]:
    """从交易日序列中提取每月最后一个交易日"""
    if isinstance(trade_dates, pd.DataFrame):
        trade_dates = trade_dates["trade_date"]
    s = pd.Series(trade_dates).astype(str)
    months = s.str[:6]
    return s.groupby(months).max().sort_values().tolist()


def calc_avg_amount(daily_quotes: pd.DataFrame, window: int = 20) -> pd.Series:
    """计算滚动平均成交额（用于流动性过滤）"""
    return daily_quotes.groupby("ts_code")["amount"].transform(
        lambda x: x.rolling(window, min_periods=10).mean()
    )


def get_illiquid_codes(daily_quotes: pd.DataFrame, date: str,
                       threshold: float = 10000) -> list[str]:
    """获取指定日期流动性不足的股票代码"""
    day_data = daily_quotes[daily_quotes["trade_date"] == date]
    if "_avg_amount_20d" not in day_data.columns:
        return []
    return day_data[day_data["_avg_amount_20d"] < threshold]["ts_code"].tolist()
