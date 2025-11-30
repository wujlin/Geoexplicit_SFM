"""
Phase 2 配置：公共参数、路径、栅格化设置。
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# 输入
SINKS_PATH = BASE_DIR / "data" / "processed" / "sinks_phase1.csv"
# 若存在本地路网数据，可填写以避免联网拉取 OSM；None 则使用 OSM
LOCAL_ROAD_PATH = BASE_DIR / "dataset" / "geo" / "MI_road_cleaned.shp" / "MI_road_cleaned.shp.shp"
OSM_CACHE = BASE_DIR / "data" / "processed" / "osm_graph.graphml"
WORLDPOP_RASTER = None  # 可选: Path(...) 指向 worldpop tif

# 输出
OUTPUT_DIR = BASE_DIR / "data" / "processed"
WALKABLE_MASK_PATH = OUTPUT_DIR / "walkable_mask.npy"
TARGET_DENSITY_PATH = OUTPUT_DIR / "target_density.npy"
FIELD_BASELINE_PATH = OUTPUT_DIR / "field_baseline.npy"
GRAD_BASELINE_PATH = OUTPUT_DIR / "grad_baseline.npz"
SCORE_BASELINE_PATH = OUTPUT_DIR / "score_baseline.npz"
BASELINE_VIZ_PATH = OUTPUT_DIR / "baseline_field.png"
INNOVATION_MODEL_PATH = OUTPUT_DIR / "score_unet.pt"
INNOVATION_SAMPLES_PATH = OUTPUT_DIR / "score_samples.npz"
INNOVATION_VIZ_PATH = OUTPUT_DIR / "score_field_innovation.png"
FIELD_INNOVATION_PATH = OUTPUT_DIR / "field_innovation.npy"
COMPARISON_VIZ_PATH = OUTPUT_DIR / "comparison_baseline_innovation.png"

# 栅格参数
GRID_RES_M = 100.0          # 栅格分辨率（米）
BBOX_PADDING_KM = 5.0       # 在 sinks bbox 四周扩展
SINK_GAUSS_SIGMA_M = 800.0  # sink 高斯核标准差（米）
SINK_WEIGHT_KEY = "total_flow"

# OSM 拉取
NETWORK_TYPE = "drive"      # 可选: 'drive', 'walk', 'all_private'

# 计算控制
MIN_MASK_FRACTION = 0.01    # 若路网覆盖比例过低则报警
TARGET_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3857"

# 可视化/调试
VERBOSE = True
