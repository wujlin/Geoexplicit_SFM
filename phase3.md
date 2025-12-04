# Phase 3 - 轨迹仿真

## 目标

基于个体目的地采样和对应导航场，生成带 target_sink_id 的轨迹数据。

## 核心改进

**旧方案**: 全局导航场 + 固定步数

**新方案**:
1. 每个 agent 根据 OD 概率采样目的地
2. 使用对应 sink 的导航场
3. 到达终止条件

## 物理模型

Langevin 动力学 + 道路约束:
```
v_{t+1} = β·v_t + (1-β)·(V0·nav + F_wall + η)
pos_{t+1} = pos_t + v_{t+1}·dt
```

**参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| V0 | 1.5 | 基础速度 (px/step) |
| MOMENTUM | 0.7 | 动量系数 |
| NOISE_SIGMA | 0.08 | 噪声强度 |
| ARRIVAL_THRESH | 50 | 到达判定 (px, ~5km) |
| MAX_STEPS | 2000 | 最大步数 |

## 仿真流程

```python
for agent in agents:
    # 1. 采样起点 (基于距离场权重)
    origin = sample_origin(walkable_mask, distance_field)
    
    # 2. 根据起点位置采样目的地
    origin_tract = pixel_to_tract(origin)
    target_sink = sample_sink(od_sampler[origin_tract])
    
    # 3. 加载对应导航场
    nav_field = load_nav_field(target_sink)
    
    # 4. 仿真直到到达或超时
    while not arrived and steps < MAX_STEPS:
        step(nav_field)
```

## 输入/输出

**输入**:
- `nav_fields/sink_{i}.npz`: 35 个导航场
- `od_sampler.npz`: OD 采样表
- `walkable_mask.npy`: 道路 mask

**输出**:
- `trajectories.h5`:
  - `positions`: (T, N, 2)
  - `velocities`: (T, N, 2)
  - `target_sink_id`: (N,)
  - `arrival_step`: (N,)
