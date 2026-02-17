# 因子模块包
# 导入所有因子以触发 @register_factor 注册
from factors import (
    momentum, reversal, turnover, volatility, size, swim_against_tide,
    illiquidity, max_return, vwap_ratio, amount_ratio, highlow_ratio,
    price_volume_corr,
    volume_std, open_vol_rank, vwap_high_corr, highlow_close_ratio,
    ret_turn_cov, vwap_vol_cov,
)
