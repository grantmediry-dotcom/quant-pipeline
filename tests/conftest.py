"""测试 fixtures：合成行情数据"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# 确保项目根目录在 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def mock_daily_quotes():
    """合成 50 只股票 × 252 天日线数据（布朗运动价格）"""
    np.random.seed(42)
    n_stocks = 50
    n_days = 252

    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    dates = pd.bdate_range("20220101", periods=n_days).strftime("%Y%m%d").tolist()

    rows = []
    for code in codes:
        # 起始价格在 10~100 之间
        price = np.random.uniform(10, 100)
        for i, date in enumerate(dates):
            ret = np.random.normal(0.0005, 0.02)
            price *= (1 + ret)
            price = max(price, 1.0)

            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            opn = price * (1 + np.random.normal(0, 0.005))
            vol = np.random.uniform(1e5, 1e7)
            amount = vol * price / 100  # 近似成交额（千元）

            rows.append({
                "ts_code": code,
                "trade_date": date,
                "open": round(opn, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "vol": round(vol, 0),
                "amount": round(amount, 2),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


@pytest.fixture
def mock_benchmark():
    """合成基准指数 252 天"""
    np.random.seed(99)
    dates = pd.bdate_range("20220101", periods=252).strftime("%Y%m%d").tolist()
    price = 5000.0
    records = []
    for d in dates:
        price *= (1 + np.random.normal(0.0003, 0.012))
        records.append({"trade_date": d, "bench_close": round(price, 2)})
    return pd.DataFrame(records)


@pytest.fixture
def mock_factor_scores():
    """合成单截面因子得分（50 只股票，3 个因子）"""
    np.random.seed(7)
    codes = [f"{i:06d}.SZ" for i in range(1, 51)]
    df = pd.DataFrame({
        "ts_code": codes,
        "factor_a": np.random.randn(50),
        "factor_b": np.random.randn(50),
        "factor_c": np.random.randn(50),
    })
    return df
