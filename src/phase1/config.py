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

# 筛选与聚类参数
PARETO_THRESHOLD = 0.80  # 覆盖 80% 总流量
DBSCAN_EPS_KM = 1.5      # 半径，单位 km
DBSCAN_MIN_SAMPLES = 1   # 允许单点 cluster

# 坐标系
TARGET_CRS = "EPSG:4326"  # 保持经纬度；聚类时转投影或用 haversine

# 可视化
FIG_SIZE = (10, 10)
POINT_SIZE_BASE = 30
