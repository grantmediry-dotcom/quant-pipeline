"""因子单元测试：验证计算输出形状、无 Inf、NaN 比例合理"""

import numpy as np
import pandas as pd
import pytest

import factors  # noqa: F401 — 触发注册
from factor_framework.registry import FactorRegistry


REPRESENTATIVE_FACTORS = [
    "momentum_20d",
    "volatility_20d",
    "turnover_20d",
    "illiquidity_20d",
    "size_factor",
]


@pytest.fixture
def registry():
    return FactorRegistry()


class TestFactorCompute:
    """对代表性因子的 compute 输出进行基础校验"""

    @pytest.mark.parametrize("factor_name", REPRESENTATIVE_FACTORS)
    def test_output_length(self, registry, mock_daily_quotes, factor_name):
        """输出长度 == 输入行数"""
        if factor_name not in registry._factors:
            pytest.skip(f"{factor_name} 未注册")

        instance = registry.create_instance(factor_name)
        df = mock_daily_quotes.copy()
        df["ret"] = df.groupby("ts_code")["close"].pct_change()
        result = instance.compute(df)
        assert len(result) == len(df), f"输出长度 {len(result)} != 输入 {len(df)}"

    @pytest.mark.parametrize("factor_name", REPRESENTATIVE_FACTORS)
    def test_no_inf(self, registry, mock_daily_quotes, factor_name):
        """输出不含 Inf"""
        if factor_name not in registry._factors:
            pytest.skip(f"{factor_name} 未注册")

        instance = registry.create_instance(factor_name)
        df = mock_daily_quotes.copy()
        df["ret"] = df.groupby("ts_code")["close"].pct_change()
        result = instance.compute(df)
        inf_count = np.isinf(result).sum()
        assert inf_count == 0, f"包含 {inf_count} 个 Inf 值"

    @pytest.mark.parametrize("factor_name", REPRESENTATIVE_FACTORS)
    def test_nan_ratio_reasonable(self, registry, mock_daily_quotes, factor_name):
        """NaN 比例 < 50%（允许 lookback 期 NaN）"""
        if factor_name not in registry._factors:
            pytest.skip(f"{factor_name} 未注册")

        instance = registry.create_instance(factor_name)
        df = mock_daily_quotes.copy()
        df["ret"] = df.groupby("ts_code")["close"].pct_change()
        result = instance.compute(df)
        nan_ratio = result.isna().mean()
        assert nan_ratio < 0.5, f"NaN 比例 {nan_ratio:.1%} >= 50%"

    @pytest.mark.parametrize("factor_name", REPRESENTATIVE_FACTORS)
    def test_not_all_nan(self, registry, mock_daily_quotes, factor_name):
        """输出不能全是 NaN"""
        if factor_name not in registry._factors:
            pytest.skip(f"{factor_name} 未注册")

        instance = registry.create_instance(factor_name)
        df = mock_daily_quotes.copy()
        df["ret"] = df.groupby("ts_code")["close"].pct_change()
        result = instance.compute(df)
        assert result.notna().any(), "输出全部为 NaN"

    @pytest.mark.parametrize("factor_name", REPRESENTATIVE_FACTORS)
    def test_validate_passes(self, registry, mock_daily_quotes, factor_name):
        """validate 质量检验应通过"""
        if factor_name not in registry._factors:
            pytest.skip(f"{factor_name} 未注册")

        instance = registry.create_instance(factor_name)
        df = mock_daily_quotes.copy()
        df["ret"] = df.groupby("ts_code")["close"].pct_change()
        result = instance.compute(df)
        quality = instance.validate(result)
        assert quality["valid"], f"validate 失败: nan_ratio={quality['nan_ratio']:.1%}, inf={quality['inf_count']}"

    @pytest.mark.parametrize("factor_name", REPRESENTATIVE_FACTORS)
    def test_clean_removes_extremes(self, registry, mock_daily_quotes, factor_name):
        """clean 后应无 Inf，且值在合理范围"""
        if factor_name not in registry._factors:
            pytest.skip(f"{factor_name} 未注册")

        instance = registry.create_instance(factor_name)
        df = mock_daily_quotes.copy()
        df["ret"] = df.groupby("ts_code")["close"].pct_change()
        result = instance.compute(df)
        cleaned = instance.clean(result)
        assert not np.isinf(cleaned).any(), "clean 后仍有 Inf"


class TestFactorRegistry:
    """注册表基础测试"""

    def test_enabled_factors_exist(self, registry):
        enabled = registry.list_factors(enabled_only=True)
        assert len(enabled) > 0, "无启用因子"

    def test_all_factors_have_metadata(self, registry):
        for name in registry.list_factors():
            meta = registry.get_metadata(name)
            assert meta.name == name
            assert meta.display_name
            assert meta.category
            assert meta.direction in (-1, 1)

    def test_reset_eval(self, registry):
        # 手动设置 eval 值
        for name in registry.list_factors():
            meta = registry.get_metadata(name)
            meta.eval_ic_mean = 0.05
            meta.eval_ir = 1.0
            registry.update_metadata(name, meta)

        registry.reset_eval()

        for name in registry.list_factors():
            meta = registry.get_metadata(name)
            assert meta.eval_ic_mean is None
            assert meta.eval_ir is None
