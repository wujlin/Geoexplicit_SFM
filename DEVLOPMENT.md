# Phase 1 进度与改动摘要

## 已完成
- 按文档搭建结构：`config.py`、`data_loader.py`、`sink_identifier.py`、`visualizer.py`、`main_phase1.py`、`requirements.txt`。
- 数据适配：支持 `work`/`home` 11 位 tract GEOID，按七县 FIPS 过滤工作地（Monroe, Macomb, Oakland, St. Clair, Livingston, Washtenaw, Wayne）。
- 筛选与聚类：帕累托 + 流量分位过滤，DBSCAN（Haversine）聚类，输出流量加权中心、簇内点数。
- 诊断日志：原始行数、唯一工作 tract 数、过滤后流量；簇数、簇大小、Top-K 覆盖。
- 可视化：Tract 边界 + sink 散点（可选底图，默认关闭以避免网络阻塞）。

## 当前参数（可调，最新运行）
- `PARETO_THRESHOLD = 0.92`
- `FLOW_MIN_QUANTILE = 0.60`
- `DBSCAN_EPS_KM = 2.5`，`DBSCAN_MIN_SAMPLES = 2`
- `TARGET_COUNTIES`：七县 FIPS
- 输出：`data/processed/sinks_phase1.csv`，`sinks_visualization.png`

## 结果概览（最近运行）
- 帕累托后 755 条，分位过滤后 302 条
- 聚类得到 35 个核心汇，最大簇 37 个点，Top10 覆盖 ~80% 流量

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
- `geo_rasterizer` 支持本地路网（`dataset/geo/MI_road_cleaned.shp/...`），已生成 `walkable_mask.npy`、归一化的 `target_density.npy`。

## 当前策略
- 仅保留 Baseline 为 Phase2 最终产物：`field_baseline.npy`、梯度/score npz、`baseline_field.png`。
- Track B Innovation 暂停/退出 Phase2，Diffusion 计划放到 Phase4。
- Comparison 仅用于 Baseline 可视化检查。

# Phase 3 规划（数据工厂）
- 目标：基于 Baseline 导航场 + 朗之万噪声，Numba 加速批量生成合成轨迹，供 Phase4 Diffusion 训练。
- 目录：`src/phase3/`（config、core/environment & physics、simulation/spawner/engine/recorder），入口 `main_gen_data.py`。
- 输出：`data/output/trajectories.h5`（或 npz），包含大规模 `(pos, vel, mask)` 时间序列；粒子池重生模式、环形缓冲写盘。
