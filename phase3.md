# Phase 3 - Synthetic Trajectory Data Factory（更新版）

本阶段聚焦高吞吐轨迹生成，针对 1px 路网的“走廊陷阱”进行了多项修复，确保粒子能沿路网产生清晰流线。

## 核心问题与修复
- 导航场过弱/抵消：改用测地距离场负梯度，或最近邻方向（避免双线性在窄路上互相抵消）。
- 非 walkable 区无法返回：SDF 梯度用于推回路网，掉网自动恢复。
- Sink 区生成：采样域排除 dist=0 的 sink 区（valid = mask & dist>0 & dist<阈值）。
- 边界梯度污染：采用最小邻居/最近邻方向，避免中心差分被 mask 污染。

## 物理模型（带动量的 Langevin 版本）
```
v_{t+1} = β · v_t + (1-β) · (V0 · n̂(x) + F_wall + η)
pos_{t+1} = pos_t + v_{t+1} · dt
```
- 导航方向 n̂：距离场/score 的最近邻方向。
- 道路约束：若脱网，选择 4 邻域中与速度点积最大的可行走像素前进。
- 噪声：η ~ N(0, σ^2)。
- 墙/恢复：靠近边界时推回，OFF_ROAD_RECOVERY 较大。

### 建议参数（技术总结版）
| 参数 | 值 | 说明 |
|------|-----|------|
| V0 | 1.5 px/步 | 基础速度 |
| DT | 1.0 | 时间步长 |
| NOISE_SIGMA | 0.05 | 低噪声 |
| MOMENTUM β | 0.85 | 平滑轨迹 |
| WALL_PUSH | 2.0 | 墙壁斥力/恢复 |
| OFF_ROAD_RECOVERY | 5.0 | 掉网恢复力 |

## 产出
- `data/output/trajectories.h5`：positions/velocities 时间序列，粒子池重生；可视化脚本断开重生跳线。
- 性能参考（1px 路网，2000步）：到达率 ~45.7%，平均位移 ~62.4 像素，on-road ~50%，速度均值 ~2.3 px/frame。

## 模块结构
```
src/phase3/
  config.py                # 物理参数
  core/environment.py      # 加载 mask/导航场，SDF
  core/physics.py          # Numba 核心步进（导航+墙+噪声）
  simulation/spawner.py    # 采样/重生（排除 sink 区域）
  simulation/recorder.py   # 分块写入 HDF5
main_gen_data.py           # 无头生成轨迹
scripts/inspect_trajectories.py  # 统计+抽样图
scripts/plot_trajectories_clean.py # 去跳线轨迹图
```

## 使用
```
python main_gen_data.py --agents 10000 --steps 10000
python scripts/inspect_trajectories.py
python scripts/plot_trajectories_clean.py
```
