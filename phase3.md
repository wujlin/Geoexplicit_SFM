这是 Phase 3 的最新开发指引，目标是把系统转为高吞吐的数据工厂，用 Baseline 场驱动朗之万动力学，批量生成合成轨迹供后续 Diffusion 使用。

````markdown
# Geo-Explicit SFM Simulator - Phase 3: Synthetic Trajectory Data Factory

## 目标
- 利用 Phase 2 Baseline 导航场（`field_baseline.npy`）和 `walkable_mask.npy`，构建一个高性能（Numba 加速）的轨迹生成器。
- 持续生成大规模 `(State, Action)` 序列，为后续 Phase4 的 Diffusion Policy 训练提供数据。

## 产出
- `data/output/trajectories.h5`（或 `.npz`）：包含大量 Agent 的轨迹
  - 示例字段：`positions (N,T,2)`, `velocities (N,T,2)`, `masks (N,T)`，可附 `sink_id`/`origin_id`。

## 物理逻辑
- 导航力：双线性插值 `field_baseline.npy` 得到 `v_desire`。
- 墙壁力：基于 `walkable_mask` 预计算 SDF 梯度。
- 噪声：朗之万噪声，保证同一起点可生成多条略不同轨迹。
- 生命周期：固定数量粒子池；到达/超时即重生，保持算力利用率。

## 模块结构
```text
src/phase3/
├── __init__.py
├── config.py              # 物理/模拟参数
├── core/
│   ├── environment.py     # 加载 mask/field，预计算 SDF
│   └── physics.py         # Numba 核心：场插值、SDF 斥力、噪声积分
├── simulation/
│   ├── spawner.py         # OD 采样/重生逻辑
│   ├── recorder.py        # 缓冲写入 HDF5/NPZ
│   └── engine.py          # 主循环，连接 physics + spawner + recorder
└── main_gen_data.py       # 无头运行脚本（进度条）
```

## 开发步骤
1) `core/environment.py`：读取 Baseline 场、mask，计算 SDF。
2) `core/physics.py`：Numba JIT 核心；Euler-Maruyama 积分，返回到达终点的索引。
3) `simulation/spawner.py`：从 OD 表采样起点/终点，提供重生接口。
4) `simulation/recorder.py`：环形缓冲，块写入 HDF5。
5) `main_gen_data.py`：初始化粒子池（如 20k），循环若干步（如 10k），输出轨迹。

## 注意
- 仅使用 Baseline 导航场；创新 Diffusion 延后至 Phase4。
- 噪声和参数需在 `config.py` 集中管理，便于调优。
````+
