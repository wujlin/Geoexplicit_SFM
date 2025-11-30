# Phase 1 进度与改动摘要

## 已完成
- 按文档搭建结构：`config.py`、`data_loader.py`、`sink_identifier.py`、`visualizer.py`、`main_phase1.py`、`requirements.txt`。
- 数据适配：支持 `work`/`home` 11 位 tract GEOID，按七县 FIPS 过滤工作地（Monroe, Macomb, Oakland, St. Clair, Livingston, Washtenaw, Wayne）。
- 筛选与聚类：帕累托 + 流量分位过滤，DBSCAN（Haversine）聚类，输出流量加权中心、簇内点数。
- 诊断日志：原始行数、唯一工作 tract 数、过滤后流量；簇数、簇大小、Top-K 覆盖。
- 可视化：Tract 边界 + sink 散点（可选底图，默认关闭以避免网络阻塞）。

## 当前参数（可调）
- `PARETO_THRESHOLD = 0.70`
- `FLOW_MIN_QUANTILE = 0.85`（流量分位过滤）
- `DBSCAN_EPS_KM = 5.5`，`DBSCAN_MIN_SAMPLES = 3`
- `TARGET_COUNTIES`：七县 FIPS
- 输出：`data/processed/sinks_phase1.csv`，`sinks_visualization.png`

## 结果概览（最近运行）
- 过滤后高流量点：44
- 聚类得到 7 个核心汇，主簇在底特律 CBD，其他覆盖迪尔伯恩、安娜堡及北侧走廊（特洛伊/南菲尔德/诺维等），均在目标县内。

## 待办 / 风险
- 数据粒度：目前使用 `mi-tract-od-2020.csv`（tract 级、工作地裁剪区域）。若需全州/更细粒度（block）需换源并更新 `RAW_OD_PATH`。
- 需要的话可对居住地也做县域过滤，避免外域输入影响阈值。
- 参数可再微调：如想合并北侧簇，可略增 `DBSCAN_EPS_KM`；如要更细分，可减小 eps 或收紧分位阈值。

# Phase 2 进度

## Step 1: Common Infrastructure
- 新增 `src/phase2/config.py`：路径、栅格参数、OSM 网络类型、输出位置。
- 新增 `src/phase2/common/geo_rasterizer.py`：读取 Phase1 sinks、计算 bbox、OSM 下载/缓存、路网栅格化、sink 高斯核生成目标密度（可选融合 worldpop），输出 `walkable_mask.npy`、`target_density.npy`。
- 结构初始化：`src/phase2/__init__.py`、`src/phase2/common/__init__.py`、baseline/innovation 目录占位。
- 依赖更新：`requirements.txt` 增加 osmnx、rasterio、scipy。

## Step 1.5: Baseline Solver Scaffold
- 新增 `src/phase2/baseline/pde_solver.py`：有限差分扩散、梯度/score 计算，封装 `solve_field`。
- `geo_rasterizer` 支持本地路网（`dataset/geo/MI_road_cleaned.shp/...`），已生成 `walkable_mask.npy`（598x923，mean≈0.425）和归一化的 `target_density.npy`。

## 待办（Phase2 后续）
- Baseline: 编写 `main_phase2_baseline.py`，读取 mask/density，调用 `solve_field`，保存 `field_baseline.npy`（以及梯度/score npz），可加简易可视化。 ✅ 已完成，输出 `baseline_field.png`。
- Innovation: dataset/network/trainer + `main_phase2_innovation.py`。✅ 已完成，当前使用静态输入（mask+density），全图前向输出 `field_innovation.npy`，训练配置写入同名 `.json`；稀疏可视化 `score_field_innovation.png`。
- Comparison: 并排流线可视化。✅ `main_phase2_compare.py` 生成 `comparison_baseline_innovation.png` 及指标 JSON（方向余弦、角度、差分）。
