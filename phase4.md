# Phase 4 - Diffusion Policy

## 目标

学习条件分布 p(action | obs)，其中 obs 包含位置、速度和导航方向。

## 方法

DDPM + 1D Conditional UNet

**输入**: obs = [pos, vel, nav_direction] (6 维 × history)

**输出**: action = velocity (2 维 × future steps)

## 架构

```
src/phase4/
├── config.py
├── data/
│   ├── dataset.py      # 滑窗 Dataset
│   └── normalizer.py   # ZScore + IdentityNormalizer
├── model/
│   └── unet1d.py       # 1D UNet
├── diffusion/
│   └── scheduler.py    # DDPM/DDIM
├── train.py
└── inference.py
```

## 训练

```bash
python src/phase4/train.py --epochs 50 --batch_size 16384
```

## 推理

MPC 模式: 预测 8 步，执行 1 步

## 验证指标

| 指标 | 定义 |
|------|------|
| Pred vs Nav cos_sim | 预测方向与导航方向的余弦相似度 |
| Pred vs GT cos_sim | 预测与真实的余弦相似度 |
| Approaching Rate | 朝向目的地移动的比例 |

## 已修复问题

- IdentityNormalizer: nav_direction 不使用 ZScore，避免角度扭曲
