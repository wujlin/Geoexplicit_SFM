# Phase 2 - 导航场生成（加权引力场）

本阶段采用“加权引力场叠加”方案，避免标量扩散场梯度衰减、单源距离场丢失权重的问题。

## 核心思路
- 每个 sink 产生一个引力场，强度与流量权重成正比、与距离成反比：  
  `phi(x) = sum_i w_i / (1 + alpha * d_i(x))`，`alpha = 0.05`。  
  导航方向 `v(x) = grad(phi) / |grad(phi)|`。
- 使用方向场（score）作为输出：`score_baseline.npz`（score_y/score_x）。
- 非 walkable 区可结合 mask/SDF 做约束。

## 输入/输出
- 输入：Phase1 `sinks_phase1.csv`；行走区域掩膜 `walkable_mask.npy`；可选 WorldPop/target_density。
- 输出：`score_baseline.npz`（方向场），供 Phase3 直接使用。

## 目录
```
data/processed/
  sinks_phase1.csv
  walkable_mask.npy
  target_density.npy
  score_baseline.npz   # 导航方向场
src/phase2/
  config.py
  common/geo_rasterizer.py   # 栅格化/密度
  baseline/pde_solver.py     # 势场/梯度计算
main_phase2_baseline.py      # 生成导航场
```

## 开发要点
1) 栅格化：生成 walkable_mask、target_density。  
2) 势场：按加权引力公式计算 phi，取梯度归一化为导航方向。  
3) 可视化：流线或箭头检查方向指向主要汇，梯度不应快速衰减。
