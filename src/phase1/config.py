"""
Phase 1 配置：集中管理路径与关键阈值。
可按需调整 eps/min_samples 等参数。
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# 数据路径（对齐当前仓库 dataset 目录）
RAW_OD_PATH = BASE_DIR / "dataset" / "od_flow" / "mi-tract-od-2020.csv"
RAW_SHAPE_PATH = BASE_DIR / "dataset" / "geo" / "mi-tracts-demo-work" / "mi-tracts-demo-work.shp.shp"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_SINKS_CSV = OUTPUT_DIR / "sinks_phase1.csv"
OUTPUT_VIZ = OUTPUT_DIR / "sinks_visualization.png"

# 筛选与聚类参数（目标 ~30-60 sinks）
PARETO_THRESHOLD = 0.92   # 覆盖 92% 总流量
FLOW_MIN_QUANTILE = 0.60  # 略放宽分位
FLOW_MIN_VALUE = None     # 或者指定绝对值（S000），None 则按分位数
DBSCAN_EPS_KM = 2.5       # 适中半径，避免极大簇
DBSCAN_MIN_SAMPLES = 2    # 至少 2 点成簇，减少单点簇

# 坐标系
TARGET_CRS = "EPSG:4326"  # 保持经纬度；聚类时转投影或用 haversine
PROJECTED_CRS = "EPSG:3857"  # 用于计算质心/可视化

# 可视化
FIG_SIZE = (10, 10)
POINT_SIZE_BASE = 30
USE_BASEMAP = False  # 无网络时关闭底图

# 日志
LOG_TOP_K = 10  # 打印前 K 个簇的覆盖情况

# 只关注的县（前 5 位 FIPS）
TARGET_COUNTIES = ["26115", "26099", "26125", "26147", "26093", "26161", "26163"]  # Monroe, Macomb, Oakland, St. Clair, Livingston, Washtenaw, Wayne
