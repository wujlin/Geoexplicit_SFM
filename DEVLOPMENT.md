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
- 新增 `src/phase2/baseline/pde_solver.py`：加权引力场叠加，`phi(x)=sum_i w_i/(1+alpha*d_i)`，梯度归一化为导航方向 (score)。
- `geo_rasterizer` 支持本地路网（`dataset/geo/MI_road_cleaned.shp/...`），已生成 `walkable_mask.npy`、`target_density.npy`。

## 当前策略
- 仅保留 Baseline 为 Phase2 最终产物：导航方向场 `score_baseline.npz`（加权引力梯度），可视化 `baseline_field.png`。
- Track B Innovation 暂停/退出 Phase2，Diffusion 计划放到 Phase4。
- Comparison 仅用于 Baseline 可视化检查。

# Phase 3 规划（数据工厂）
- 目标：基于 Baseline 导航场（方向场）+ 带动量/约束的 Langevin，Numba 加速批量生成合成轨迹，供 Phase4 Diffusion 训练。
- 目录：`src/phase3/`（config、core/environment & physics、simulation/spawner/engine/recorder），入口 `main_gen_data.py`。
- 输出：`data/output/trajectories.h5`，包含大规模 `(pos, vel)` 时间序列；粒子池重生模式、环形缓冲写盘。
- 物理内核（技术总结版）：带动量 Langevin + 道路约束，导航方向采用最近邻/距离场梯度；掉网恢复用 SDF 推回。参考参数：`V0≈1.5 px/步`，`DT=1.0`，`NOISE_SIGMA≈0.05`，`MOMENTUM≈0.85`，`WALL_PUSH≈2.0`，`OFF_ROAD_RECOVERY≈5.0`。可视化脚本支持断开重生跳线。

## Phase 3 问题诊断与修复（近期）
- 导航场误用：曾加载 `field_baseline.npy`（标量），现改为方向场（score 或距离场负梯度）。
- 场强过弱：原 score 场多数像素模长极小，改用测地距离场负梯度/最近邻方向。
- 掉网/积灰：非 walkable 区导航为零，加入 SDF 恢复力，非 walkable 区推回路网。
- Sink 生成：spawner 排除 dist=0 的 sink 区（有效采样域限制）。
- 插值抵消：1px 路网双线性可能抵消，改用最近邻/最小邻居方向。
- 边界梯度污染：中心差分受 mask 影响，改用最小邻居距离方向。
- 当前指标（1px 路网，2000步）：到达率 ~45.7%，平均位移 ~62.4 像素，on-road ~50%，速度均值 ~2.3 px/frame。
- 后续可选：膨胀 walkable mask、调整恢复力、使用 8 邻域导航。

# Phase 4 摘要（Diffusion Policy）
- 目标：基于 Phase3 轨迹 (`data/output/trajectories.h5`) 训练条件扩散策略。
- 目录：`src/phase4/`（config、data/{dataset,normalizer}、model/unet1d、diffusion/scheduler、train.py、inference.py）。
- Dataset：`TrajectorySlidingWindow`，obs=过去2帧位置+速度，action=未来8帧速度。
- 模型：条件 1D UNet（timestep embedding + 全局条件），噪声预测 MSE，EMA 平滑。
- 调度：`scheduler.py` 实现 DDPM/可选 DDIM。
- 使用：
  - 训练：`python src/phase4/train.py --epochs 100 --batch_size 256`
  - 推理：`python src/phase4/inference.py --num_agents 20 --max_steps 500`（MPC：预测8步，执行1步）
