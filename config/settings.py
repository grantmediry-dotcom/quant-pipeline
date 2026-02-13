"""全局配置"""
import os

# Tushare 配置
TUSHARE_TOKEN = "09851f85493b02bb928aa1b9e4d7a4ff1c2b7dc7d29bbde2fa57c22b"

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 回测参数
INDEX_CODE = "852000.SH"       # 中证1000
BENCHMARK_CODE = "000852.SH"   # 中证1000指数（行情代码）
START_DATE = "20200101"
END_DATE = "20241231"
REBALANCE_FREQ = "M"           # 月度调仓
TOP_N = 50                     # 每期选股数量

# Tushare 限速（免费账户每分钟200次）
API_DELAY = 0.35               # 每次请求间隔（秒）

# 因子配置已迁移至 factor_framework + factors/ 模块（装饰器注册模式）
# 新增因子只需在 factors/ 下创建文件，用 @register_factor 注册即可
