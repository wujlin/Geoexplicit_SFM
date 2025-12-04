# Phase 2 - 导航场生成

## 目标

为每个 sink 生成独立的导航场，并构建 OD 采样表。

## 方法

### 1. 单 sink 导航场

对每个 sink i，求解 Eikonal 方程得到距离场，导航方向为负梯度:
```
nav_i(x) = -∇d_i / |∇d_i|
```

### 2. OD 采样表

将 tract-level OD 聚合到 sink-level:
```
P(sink_j | home_tract) ∝ Σ OD[home_tract → work_tract_k]
                          where work_tract_k 属于 sink_j
```

## 输入/输出

**输入**:
- `sinks_phase1.csv`: 35 个 sink
- `walkable_mask.npy`: 道路 mask
- `semcog_tract_od_intra_2020.csv`: OD 数据
- tract 地理数据

**输出**:
- `nav_fields/sink_{i}.npz`: 35 个导航场 (~550 MB)
- `od_sampler.npz`: OD 采样概率表 (~200 KB)

## 实现步骤

1. 构建 tract → pixel 映射
2. 构建 tract → sink 映射
3. 聚合 OD 到 sink 级别
4. 为每个 sink 计算导航场
