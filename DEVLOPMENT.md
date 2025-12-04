# 开发进度与改动摘要

> 最后更新: 2024-12-03

---

## 整体架构

```
Phase 1: Sink Identification
    ↓ sinks_phase1.csv (35 sinks + total_flow)
Phase 2: Navigation Field Generation  
    ↓ nav_fields/ (35 个独立导航场)
    ↓ od_sampler.npz (OD 采样概率表)
Phase 3: Trajectory Simulation
    ↓ trajectories.h5 (带 target_sink_id)
Phase 4: Diffusion Policy Learning
    ↓ trained model
验证: OD 分布相关性 + 到达率 + 道路保持率
```

---

## Phase 1: Sink Identification ✅

**状态**: 完成

**方法**: DBSCAN 聚类 OD 流量热点

**参数**:
- `PARETO_THRESHOLD = 0.92`
- `DBSCAN_EPS_KM = 2.5`
- `DBSCAN_MIN_SAMPLES = 2`

**输出**: `data/processed/sinks_phase1.csv` (35 sinks)

---

## Phase 2: Navigation Field Generation 🔄

**状态**: 需要修改

**当前问题**: 只生成单一全局导航场

**改进**:
1. 为每个 sink 单独计算导航场
2. 构建 OD 采样表 (tract → sink 概率)

**新输出**:
- `data/processed/nav_fields/sink_{i}.npz`
- `data/processed/od_sampler.npz`

---

## Phase 3: Trajectory Simulation 🔄

**状态**: 需要修改

**当前问题**: 全局导航场 + 固定步数

**改进**:
1. 个体目的地采样 (基于 OD 分布)
2. 使用对应 sink 的导航场
3. 到达终止条件

**物理模型**: `v_{t+1} = β·v_t + (1-β)·(V0·nav + η)`

**参数**: V0=1.5, MOMENTUM=0.7, NOISE=0.08

**输出**: `trajectories.h5` (含 target_sink_id)

---

## Phase 4: Diffusion Policy ✅

**状态**: 架构完成，待数据更新后重训

**方法**: DDPM + 1D UNet

**输入**: [pos, vel, nav_direction]

**已修复**: IdentityNormalizer (避免角度扭曲)

---

## 验证体系

| 指标 | 目标 |
|------|------|
| OD 相关性 | > 0.7 |
| 到达率 | > 80% |
| 道路保持率 | > 99% |

---

## 待办

**P0**: Phase 2/3 核心改进 (个体目的地 + 多导航场)

**P1**: 重新生成数据 + 重训模型

详见 `LIMITATIONS.md`
