# GeoExplicit SFM 技术路线总结

## Phase 2: 导航场计算

### 原方案问题
1. **PDE 扩散方法**：使用离散拉普拉斯迭代平滑目标密度，但梯度场在远离 sink 处迅速衰减
2. **距离场方法**：只考虑到最近 sink 的距离，完全丢失流量权重信息

### 当前方案：加权引力场叠加

**核心思想**：每个 sink 产生一个引力场，引力与权重成正比、与距离成反比

**数学公式**：
$$\phi(x) = \sum_{i} \frac{w_i}{1 + \alpha \cdot d_i(x)}$$

其中：
- $w_i$：sink $i$ 的流量权重
- $d_i(x)$：点 $x$ 到 sink $i$ 的欧几里得距离
- $\alpha = 0.05$：衰减系数

**优势**：
1. ✅ 保留每个 sink 的流量权重
2. ✅ 梯度随距离衰减但不会消失
3. ✅ 方向始终指向"综合吸引力"最强的方向

**导航方向**：
$$\vec{v}(x) = \frac{\nabla \phi(x)}{|\nabla \phi(x)|}$$

---

## Phase 3: 轨迹仿真

### 原方案问题
1. **双线性插值**：在窄道路（1-2像素宽）上，相邻像素方向相反时插值会抵消
2. **纯势场驱动**：粒子容易在边界震荡或脱离道路
3. **无惯性**：方向变化剧烈，轨迹不平滑

### 当前方案：带动量的 Langevin 动力学 + 道路约束

**核心公式**：
$$v_{t+1} = \beta \cdot v_t + (1-\beta) \cdot \left( V_0 \cdot \hat{n}(x) + F_{wall} + \eta \right)$$

其中：
- $\beta = 0.85$：动量系数
- $V_0 = 1.5$：基础速度
- $\hat{n}(x)$：归一化导航方向（最近邻采样，非双线性插值）
- $F_{wall}$：墙壁斥力/脱网恢复力
- $\eta \sim \mathcal{N}(0, \sigma^2)$：随机扰动，$\sigma = 0.05$

**道路约束机制**：

```
if 新位置脱离道路:
    在4邻域中找到可行走的邻居
    选择与速度方向点积最大的邻居
    移动到该邻居位置
```

**关键参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| `V0` | 1.5 | 像素/步 |
| `DT` | 1.0 | 时间步长 |
| `NOISE_SIGMA` | 0.05 | 低噪声 |
| `MOMENTUM` | 0.85 | 高动量（平滑轨迹）|
| `WALL_PUSH` | 2.0 | 墙壁斥力 |
| `OFF_ROAD_RECOVERY` | 5.0 | 脱网恢复力 |

**性能指标**：
- 道路保持率：100%
- 平均速度：0.7 px/step
- 方向变化均值：18.4°
- 势能增加比例：96.8%（粒子向 sink 移动）

---

## Phase 4: Diffusion Policy

### 方法
使用 DDPM (Denoising Diffusion Probabilistic Models) 学习条件动作分布 $p(a|o)$。

**模型架构**：1D Conditional UNet
- 输入：带噪动作序列 $(B, T, 2)$
- 条件：历史状态 $(B, history \times 4)$ + 时间步嵌入
- 输出：预测噪声 $(B, T, 2)$

**训练目标**：
$$\mathcal{L} = \mathbb{E}_{t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(a_t, t, o) \|^2 \right]$$

**推理**（DDIM 加速）：
- 训练时 100 步扩散
- 推理时 20 步采样
- MPC 模式：预测 8 步，执行 1 步

---

## 文件结构

```
src/
├── phase2/baseline/pde_solver.py   # 场计算
│   ├── compute_potential_field()   # 加权引力场
│   └── compute_potential_navigation()  # 梯度→导航方向
├── phase3/
│   ├── config.py                   # 物理参数
│   └── core/physics.py             # Numba 加速的粒子步进
│       └── step_kernel()           # 动量 + 道路约束
└── phase4/
    ├── train.py                    # DDPM 训练
    ├── inference.py                # 闭环推理
    ├── diffusion/scheduler.py      # DDPM/DDIM
    └── model/unet1d.py             # 1D UNet
```
