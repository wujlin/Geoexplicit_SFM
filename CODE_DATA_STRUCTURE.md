# 代码与数据结构文档

> **目的**：记录项目中所有关键数据结构、坐标约定、文件格式，避免重复踩坑。

---

## 1. 全局坐标约定

### 1.1 坐标系统

```
坐标系: 图像坐标系 (row, col) = (y, x)

  ┌───────────────────────► x (col, 第2维)
  │
  │   (0,0)────────────────►
  │     │
  │     │   地图区域
  │     │
  │     ▼
  │
  ▼ y (row, 第1维)
```

**核心约定**：
- **所有 2D 数组**：第一维是 y (row)，第二维是 x (col)
- **所有位置向量**：`[y, x]` 顺序
- **所有速度向量**：`[vy, vx]` 顺序
- **导航场方向**：`[nav_y, nav_x]` 顺序

### 1.2 索引方式

```python
# 正确的索引方式
value = field[y, x]           # 2D 场
value = field[:, y, x]        # (C, H, W) 格式的场

# 位置向量
pos = np.array([y, x])        # 形状 (2,)
iy, ix = int(pos[0]), int(pos[1])
value = field[iy, ix]
```

---

## 2. 核心数据文件格式

### 2.1 轨迹数据 `data/output/trajectories.h5`

```
HDF5 文件结构:
├── positions:     (T, N, 2) float32  # [y, x] 顺序
├── velocities:    (T, N, 2) float32  # [vy, vx] 顺序
└── destinations:  (T, N) int32       # 目标 sink ID

维度说明:
- T = 10000: 时间步数
- N = 10000: agent 数量
- 2: 空间维度 [y, x]
```

**⚠️ 关键语义 (2025-12-04 验证)**：

```
vel[t] = pos[t] - pos[t-1]    # 到达 pos[t] 的位移（不是离开！）
vel[t+1] = pos[t+1] - pos[t]  # 从 pos[t] 到 pos[t+1] 的位移

即: vel[t] 描述的是 "如何到达 pos[t]"，而不是 "从 pos[t] 出发"
```

这是因为 `recorder.collect(pos, vel)` 在 `step_kernel` 执行**之后**调用，
此时 `vel` 已经被更新为 `(new_pos - old_pos) / dt`。

**验证结果**：
- `vel[t] == pos[t+1] - pos[t]`: 只有 0.9% 一致 ❌
- `vel[t+1] == pos[t+1] - pos[t]`: **99.7% 一致** ✅

### 2.2 导航场 `data/processed/nav_fields/` ⚠️ 重要

```
目录结构:
nav_fields/
├── nav_fields_index.json     # 索引文件
├── nav_field_000.npz         # sink 0 的导航场
├── nav_field_001.npz         # sink 1 的导航场
└── ...

单个 .npz 文件内容:
├── nav_y: (H, W) float32         # y 方向分量
├── nav_x: (H, W) float32         # x 方向分量
└── distance_field: (H, W) float32  # ⚠️ 到该 sink 的距离场

加载后组合为: (2, H, W) 格式，即 [nav_y, nav_x]
```

**⚠️ 关键区分（不要混淆）**：

| 文件 | 含义 | 用途 |
|------|------|------|
| `distance_field.npy` | 到**最近** sink 的距离 | 全局可达性分析 |
| `nav_field_XXX.npz['distance_field']` | 到**指定** sink XXX 的距离 | Phase 3 评估、目标导航 |

**Phase 3 仿真逻辑**：
- 每个 agent 有明确的 `destination` (目标 sink ID)
- 使用 `nav_field_{destination}.npz` 中的导航场和距离场
- 评估时必须用到**目标 sink** 的距离，而非最近 sink

**加载代码**：
```python
def load_nav_fields(nav_fields_dir):
    """加载所有个体导航场（含距离场）"""
    nav_fields = {}
    dist_fields = {}
    index = json.load(open(nav_fields_dir / "nav_fields_index.json"))
    for sink_id in index["sink_ids"]:
        data = np.load(nav_fields_dir / f"nav_field_{sink_id:03d}.npz")
        nav_fields[sink_id] = np.stack([data["nav_y"], data["nav_x"]], axis=0)  # (2, H, W)
        dist_fields[sink_id] = data["distance_field"]  # (H, W)
    return nav_fields, dist_fields
```

### 2.3 全局导航场 `data/processed/nav_baseline.npz`

```
文件内容:
├── nav_y: (H, W) float32     # y 方向分量
├── nav_x: (H, W) float32     # x 方向分量
├── pot:   (H, W) float32     # 势场（可选）
└── dist:  (H, W) float32     # 距离场（可选）

使用方式:
nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)  # (2, H, W)
```

### 2.4 其他关键文件

| 文件路径 | 格式 | 说明 |
|---------|------|------|
| `data/processed/walkable_mask.npy` | `(H, W) bool` | 可行走区域掩码 |
| `data/processed/distance_field.npy` | `(H, W) float32` | 到道路的有符号距离 |
| `data/processed/target_density.npy` | `(H, W) float32` | 目标密度分布 |
| `data/processed/sinks_phase1.csv` | CSV | sink 信息：id, y, x, population |
| `data/processed/pixel_to_sink.npy` | `(H, W) int32` | 每个像素最近的 sink ID |

---

## 3. Phase 4 Diffusion Policy 数据结构

### 3.1 数据集 `TrajectorySlidingWindow`

```python
# __getitem__ 返回格式
{
    "obs": torch.Tensor,     # (history, 6) = [pos_y, pos_x, vel_y, vel_x, nav_y, nav_x]
    "action": torch.Tensor,  # (future, 2) = [vel_y, vel_x]
    "agent": int,
    "t0": int,
    "dest": int,
}

# 维度说明
history = 2   # 观测帧数
future = 8    # 预测帧数
obs_dim = history * 6 = 12  # 展平后的观测维度
act_dim = 2   # 动作维度

# 数据对应关系
# 假设 t0 = 1000, history = 2, future = 8:
obs[0] = [pos[1000], vel[1000], nav_dir[1000]]   # 第 1 帧
obs[1] = [pos[1001], vel[1001], nav_dir[1001]]   # 第 2 帧
action[0:8] = vel[1002:1010]                      # 未来 8 帧速度
```

### 3.2 模型输入/输出

```python
# UNet1D 模型
model = UNet1D(
    obs_dim=12,           # 观测维度 (history * 6)
    act_dim=2,            # 动作维度
    base_channels=128,    # 基础通道数
    cond_dim=64,          # 条件嵌入维度
    time_dim=64,          # 时间嵌入维度
)

# 前向传播
# x: (B, act_dim, T) = (B, 2, future)  注意: Conv1D 期望通道在中间
# timesteps: (B,) int64
# global_cond: (B, obs_dim) = (B, 12)
output = model(x, timesteps, global_cond)  # (B, 2, future)
```

### 3.3 归一化器

```python
# ObsNormalizer: 分别归一化 position, velocity, nav_direction
obs_normalizer = ObsNormalizer(mode="zscore", include_nav=True)
# - pos: ZScoreNormalizer
# - vel: ZScoreNormalizer  
# - nav: IdentityNormalizer(scale=2.0)  # 保持方向角度不变！

# ActionNormalizer: 归一化 velocity
action_normalizer = ActionNormalizer(mode="zscore")

# 使用方式
obs_normed = obs_normalizer.transform(obs)          # (B, history, 6)
action_normed = action_normalizer.transform(action)  # (B, future, 2)
```

### 3.4 Checkpoint 结构

```python
checkpoint = {
    "epoch": int,
    "model_state_dict": dict,
    "ema_state_dict": dict,           # EMA 权重（推理时用这个）
    "optimizer_state_dict": dict,
    "loss": float,
    "action_normalizer": dict,
    "obs_normalizer": dict,
    "config": {
        "history": 2,
        "future": 8,
        "obs_dim": 12,
        "act_dim": 2,
        "base_channels": 128,
        "cond_dim": 64,
        "time_dim": 64,
        "num_diffusion_steps": 100,
        "cfg_dropout_prob": 0.1,      # CFG 训练时 condition 丢弃概率
    }
}
```

### 3.5 Classifier-Free Guidance (CFG)

**背景**：原始模型对 condition 不敏感（训练后比训练前更不敏感），这是条件扩散模型的常见问题。

**解决方案**：CFG (Classifier-Free Guidance)

**训练时** (`src/phase4/train.py`):
```python
# 10% 概率丢弃 condition
if self.cfg_dropout_prob > 0:
    dropout_mask = torch.rand(B, device=self.device) < self.cfg_dropout_prob
    global_cond = global_cond * (~dropout_mask).unsqueeze(-1).float()
```

**推理时** (`src/phase4/diffusion/scheduler.py: DDIMScheduler.sample_cfg`):
```python
# CFG 公式
eps_uncond = model(x, t, zeros_condition)  # 无条件预测
eps_cond = model(x, t, condition)           # 有条件预测
eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

**参数建议**:
- `cfg_dropout_prob`: 0.1 (训练时)
- `guidance_scale`: 2.0 ~ 5.0 (推理时，越大 condition 影响越强)

**使用方式**:
```bash
# 训练（默认启用 CFG）
python src/phase4/train.py --cfg_dropout 0.1

# 推理（使用 CFG）
python src/phase4/inference.py --guidance_scale 3.0
```

---

## 4. Phase 3 仿真数据流

### 4.1 核心物理循环

```python
# src/phase3/core/physics.py: step_kernel
def step_kernel(pos, vel, active, field, sdf, mask, dt, noise_sigma, v0, ...):
    """
    输入/输出:
    - pos: (N, 2) [y, x] 位置，原地更新
    - vel: (N, 2) [vy, vx] 速度，原地更新
    - field: (2, H, W) 导航场 [field_y, field_x]
    
    更新逻辑:
    1. 从 field 采样导航方向
    2. 添加噪声和动量
    3. 墙壁推力
    4. 更新位置: pos += vel
    5. 边界约束
    """
```

### 4.2 Recorder 数据收集

```python
# src/phase3/simulation/recorder_v2.py
class TrajRecorderV2:
    def collect(self, pos, vel, dest):
        """
        收集每一帧数据
        pos: (N, 2) [y, x]
        vel: (N, 2) [vy, vx]  # 这是更新后的速度
        dest: (N,) 目标 sink ID
        """
```

**⚠️ 重要**：`vel` 记录的是 **更新后** 的速度，即 `pos[t+1] - pos[t]` 的预期值。

---

## 5. OD (Origin-Destination) 分析

### 5.1 OD 矩阵计算

```python
# 从轨迹提取 OD 对
# 一次 "trip" 定义为: agent 的 destination 发生变化

with h5py.File("trajectories.h5", "r") as f:
    dest = f["destinations"][:]  # (T, N)
    
for agent in range(N):
    agent_dest = dest[:, agent]
    for t in range(1, T):
        if agent_dest[t] != agent_dest[t-1]:
            # 发生了 respawn
            origin = agent_dest[t-1]  # 上一个目的地（到达的地方）
            destination = agent_dest[t]  # 新目的地
            # 记录 OD 对: (origin, destination)
```

### 5.2 Pixel-to-Sink 映射

```python
# data/processed/pixel_to_sink.npy: (H, W) int32
# 每个像素对应最近的 sink ID

# 计算方式
for sink_id, sink_info in sinks.items():
    dist = np.sqrt((Y - sink_y)**2 + (X - sink_x)**2)
    # 更新最近 sink
```

---

## 6. 常见错误汇总

### 6.1 坐标顺序错误

```python
# ❌ 错误
value = field[x, y]
pos = np.array([x, y])

# ✅ 正确
value = field[y, x]
pos = np.array([y, x])
```

### 6.2 HDF5 维度错误

```python
# ❌ 错误：假设格式是 (N, T, 2)
agent_pos = positions[agent_idx, :, :]

# ✅ 正确：格式是 (T, N, 2)
agent_pos = positions[:, agent_idx, :]
```

### 6.3 导航场加载错误

```python
# ❌ 错误：直接 stack 顺序错误
nav_field = np.stack([nav_x, nav_y], axis=0)

# ✅ 正确：[nav_y, nav_x] 顺序
nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)
```

### 6.4 Conv1D 输入格式错误

```python
# ❌ 错误：直接传入 (B, T, C)
output = model(action)  # action: (B, future, 2)

# ✅ 正确：转换为 (B, C, T)
action_input = action.permute(0, 2, 1)  # (B, 2, future)
output = model(action_input, timesteps, condition)
output = output.permute(0, 2, 1)  # 转回 (B, future, 2)
```

### 6.5 归一化器 nav_direction 处理

```python
# ❌ 错误：对 nav_direction 使用 ZScore（会扭曲角度）
nav_normalizer = ZScoreNormalizer()

# ✅ 正确：使用 IdentityNormalizer 保持方向
nav_normalizer = IdentityNormalizer(scale=2.0)  # 仅缩放到 [-2, 2]
```

---

## 7. 文件路径约定

```
项目根目录/
├── data/
│   ├── output/
│   │   ├── trajectories.h5              # 主轨迹数据
│   │   ├── trajectories.npz             # 轨迹备份
│   │   ├── valid_indices.npy            # 有效样本索引
│   │   ├── phase4_checkpoints/
│   │   │   ├── best.pt                  # 最佳模型
│   │   │   ├── latest.pt                # 最新模型
│   │   │   └── training_history.npz
│   │   ├── validation/                  # 验证输出
│   │   │   ├── reports/
│   │   │   ├── figures/
│   │   │   └── metrics/
│   │   └── figures/
│   └── processed/
│       ├── walkable_mask.npy
│       ├── distance_field.npy
│       ├── target_density.npy
│       ├── sinks_phase1.csv
│       ├── pixel_to_sink.npy
│       ├── nav_baseline.npz             # 全局导航场
│       └── nav_fields/                  # 个体导航场目录
│           ├── nav_fields_index.json
│           └── nav_field_XXX.npz
├── src/
│   ├── phase1/                          # Sink 识别
│   ├── phase2/                          # 导航场生成
│   ├── phase3/                          # 轨迹仿真
│   └── phase4/                          # Diffusion Policy
└── scripts/
    └── validation/                      # 验证脚本
```

---

## 8. 调试检查清单

### 新增代码时必查：

- [ ] 坐标顺序是否是 `[y, x]`？
- [ ] HDF5 索引是否是 `[t, agent, :]`？
- [ ] 导航场 stack 顺序是否是 `[nav_y, nav_x]`？
- [ ] Conv1D 输入是否需要 permute？
- [ ] 归一化器是否正确加载？
- [ ] **距离场用对了吗？**（见下）

### ⚠️ 距离场使用检查（易错点）

```python
# ❌ 错误：使用全局距离场评估"到目标 sink 的距离"
dist_field = np.load('data/processed/distance_field.npy')  # 到最近 sink
dist_to_target = dist_field[y, x]  # 这是到最近 sink，不是到目标 sink！

# ✅ 正确：使用目标 sink 的独立距离场
target_sink = dest[t, agent]
sink_dist = np.load(f'data/processed/nav_fields/nav_field_{target_sink:03d}.npz')['distance_field']
dist_to_target = sink_dist[y, x]  # 这才是到目标 sink 的距离
```

**何时用哪个？**
| 场景 | 用哪个距离场 |
|------|-------------|
| 检查是否在任意 sink 附近 | `distance_field.npy` |
| 评估 Phase 3 轨迹质量 | `nav_field_XXX.npz['distance_field']` |
| 计算到特定目的地的距离 | `nav_field_XXX.npz['distance_field']` |

### 数据一致性检查：

```python
# 验证 velocity 与 position 一致
assert np.allclose(pos[t+1] - pos[t], vel[t], atol=0.1)

# 验证导航方向是单位向量
assert np.allclose(np.linalg.norm(nav_dir), 1.0, atol=0.01)

# 验证 obs 维度
assert obs.shape == (history, 6)  # [pos(2) + vel(2) + nav(2)]
```

---

*最后更新: 2025-12-05*

---

## 9. 已知问题与解决方案

### 9.1 轨迹数据中的震荡问题

**问题描述**：
- 大量 agent 在到达目的地附近后会来回震荡
- 震荡样本的速度方向不一致，但速度大小正常
- 这导致 Diffusion Policy 学到"随机方向"而非"朝目的地前进"

**诊断方法**：
```python
# 检查方向一致性
for t in range(10, 1000):
    v1, v2 = vel[t-1], vel[t]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if cos_angle < -0.5:  # 方向反转
        print(f"Oscillation at t={t}")
```

**解决方案**：
1. 在数据过滤时增加方向一致性检查
2. 或者在仿真时让到达目的地的 agent 停止/消失
3. 或者使用 respawn 前的轨迹段

### 9.2 Diffusion Policy 模型对 Condition 不敏感 (核心问题)

**问题诊断结果 (2025-01):**

1. **模型输出几乎是常数**
   - 无论输入什么 condition，模型输出都约为 `[1.0, 1.0]` (归一化空间)
   - 反归一化后约为 `[0.86, 0.74]` (固定速度向量)

2. **训练使模型更不敏感**
   - 随机初始化模型: condition 变化引起输出差异 = 0.17
   - 训练后模型: condition 变化引起输出差异 = 0.11
   - **训练使模型更不敏感了！**

3. **但模型确实在"工作"**
   - 对不同的输入噪声有不同响应
   - 预测的噪声 std ≈ 0.7-0.8 (合理范围)
   - 训练 loss 稳定下降到 0.30

4. **根本原因分析 (2025-12-04 确认)**
   - ~~扩散模型学会了预测"平均噪声"，忽略 condition~~
   - **真正原因: ResBlock1D 的 condition 注入使用简单加法 `h = h + cond_proj(cond)`**
   - 简单加法太弱，模型学会忽略 condition (因为这样梯度更小)
   - CFG 训练虽然正确实现，但因为注入方式太弱，无法生效

**已实施的解决方案 (v2):**

使用 **FiLM (Feature-wise Linear Modulation)** 替代简单加法:

```python
# 之前 (错误): 简单加法
h = h + self.cond_proj(cond).unsqueeze(-1)

# 现在 (正确): FiLM
film_params = self.cond_proj(cond)  # 输出 gamma + beta
gamma, beta = film_params.chunk(2, dim=-1)
h = gamma * h + beta  # 乘法 + 加法，condition 无法被忽略
```

**为什么 FiLM 更好:**
- 如果 gamma=0，所有特征都会被杀死 → 模型被迫学习有意义的 gamma
- 乘法比加法更强，condition 信息更难被忽略
- 这是 conditional diffusion 的标准做法 (StyleGAN, DDPM, etc.)

**训练建议:**
```bash
# 使用 FiLM 重新训练
python src/phase4/train.py --cfg_dropout 0.15  # 可适当增大 dropout
```

### 9.3 cos_sim 评估指标

**正确计算方式**：
```python
# 预测与 GT 的方向一致性
pred = model_output[0]  # 第一步预测
gt = ground_truth[0]    # 第一步 GT
cos_sim = np.dot(pred, gt) / (np.linalg.norm(pred) * np.linalg.norm(gt))

# 预期值：
# - 随机预测: cos_sim ≈ 0
# - 好的模型: cos_sim > 0.6
# - 优秀模型: cos_sim > 0.8
```

### 9.4 DDIM clip_sample_range 问题 (2025-12-04 发现)

**问题**：
- 默认 `clip_sample_range=1.0`
- 但归一化后数据范围是 `[-3, 3]`，64% 的数据超出 `[-1, 1]`
- 导致推理时预测被截断

**解决**：
- 已修改 `scheduler.py` 中 `clip_sample_range` 从 1.0 改为 5.0

---

## 10. 诊断记录 (避免重复思考)

### 10.1 数据质量验证 (已确认 OK)

| 检查项 | 结果 | 说明 |
|-------|------|-----|
| nav 与 GT action 的 cos_sim | 0.67~0.82 | 数据本身是好的 |
| obs velocity 与 action 的连续性 | 0.71 | 时序一致性好 |
| 静止样本比例 | 0.11% | 可忽略 |
| 归一化后数据分布 | mean≈0, std≈1 | 正确 |

**结论**：数据没问题，问题在模型训练。

### 10.2 模型问题诊断 (已确认)

**现象**：
- 模型输出几乎是常数 `[~1.7, ~-0.7]`（归一化空间）
- 不同 condition 的输出差异极小（std < 0.01）
- cos_sim(pred, GT) ≈ 0（随机水平）

**根因分析**：

1. **condition 影响被忽略**
   - 测试：相同 x_t + 不同 condition → 差异 0.032
   - 测试：不同 x_t + 相同 condition → 差异 0.207
   - 比值 0.15，说明模型主要依赖 x_t

2. **加法注入太弱**
   - `h = h + cond_proj(cond)` 容易被后续层"平均掉"
   - 训练时梯度会倾向于忽略 condition

3. **CFG 训练无效**
   - 虽然实现了 10% dropout
   - 但因为注入方式太弱，模型本身就不依赖 condition

### 10.3 已实施的修复

| 修复项 | 文件 | 内容 |
|-------|------|-----|
| clip_sample_range | `scheduler.py` | 1.0 → 5.0 |
| FiLM 注入 | `unet1d.py` | `h = gamma * h + beta` |

### 10.4 FiLM vs 加法注入对比

| 指标 | 加法注入（训练后） | FiLM（随机初始化） |
|-----|------------------|------------------|
| condition 差异 | 0.032 | 0.108 |
| input 差异 | 0.207 | 0.138 |
| 比值 | 0.15 | **0.78** |

**结论**：FiLM 显著提升 condition 影响力，需要重新训练验证。

### 10.5 待验证事项

- [ ] FiLM 模型训练后 cos_sim 是否提升
- [ ] 最优 cfg_dropout 比例（当前 0.1，可尝试 0.15~0.2）
- [ ] 最优 guidance_scale（推理时）

### 10.6 FiLM 初始化问题 (2025-12-04 发现)

**问题**：
FiLM 公式 `h = gamma * h + beta`，如果 gamma 初始化为 0 附近，训练时 gamma 可能保持接近 0，导致：
- 特征 h 被 gamma 杀死
- 模型主要依赖 beta（加性偏置），而非 gamma（乘性调制）
- condition 敏感性很低

**诊断**：
```
训练后 FiLM 参数:
  gamma weights: mean≈0, std≈0.07, max≈0.16
  gamma bias: mean≈0, std≈0.07
  
问题：gamma bias ≈ 0 意味着 gamma ≈ 0，特征被杀死
```

**最终解决方案** (v3):

采用 **AdaLN** + **obs_scale** 组合：

1. **AdaLN**：gamma bias 初始化为 1（不是 0）
   ```python
   nn.init.ones_(self.cond_gamma.bias)  # gamma ≈ 1，特征不被杀死
   nn.init.zeros_(self.cond_beta.bias)  # beta ≈ 0
   ```

2. **obs_scale**：可学习缩放因子，初始值 2.0
   ```python
   self.obs_scale = nn.Parameter(torch.tensor(2.0))
   g_cond = self.obs_proj(global_cond) * self.obs_scale
   ```

3. **去除 cond_fuse**：time 和 obs 直接相加（标准 DDPM 做法）
   ```python
   cond = t_cond + g_cond  # 不再用 concat + Linear
   ```

**验证结果**：
| 版本 | cond/input ratio | 备注 |
|-----|-----------------|------|
| 旧加法版本 (训练后) | 0.15 | 基线 |
| FiLM gamma*h (训练后) | 0.05 | gamma→0 |
| AdaLN + obs_scale=2.0 (随机初始化) | 0.25 | 改进 |

### 10.7 AdaLN 模型评估结果 (2025-12-05)

**模型信息**：
- Checkpoint: `best.pt` (Epoch 86, Loss 0.2825)
- 架构: AdaLN + obs_scale, cfg_dropout=0.15
- 参数量: 1,059,843 (4.04 MB)

**单步预测质量**：
| 指标 | 值 | 解读 |
|------|-----|------|
| Pred vs GT cos_sim | **0.259** | 比随机(≈0)好，65.7% 为正 |
| GT vs Nav cos_sim | -0.053 | 数据本身 nav 与 GT 弱相关（震荡问题） |
| Pred vs Nav cos_sim | 0.079 | 模型对 nav 响应弱 |
| GT speed mean | 1.08 | 真实速度 |
| Pred speed mean | 0.91 | 预测速度偏小 |
| MSE (first frame) | 0.878 | - |

**条件响应测试**：
| nav 输入 | pred 均值 | cos(pred, nav) | 是否正确 |
|---------|----------|----------------|---------|
| [1, 0] (y+) | [0.11, -0.10] | 0.75 | ✓ |
| [0, 1] (x+) | [-0.04, -0.12] | **-0.95** | ✗ 反向 |
| [-1, 0] (y-) | [0.02, -0.04] | -0.42 | ✓ |
| [0, -1] (x-) | [0.02, -0.14] | 0.99 | ✓ |

**问题**：x 方向 (nav=[0,1]) 响应反向，可能是坐标约定问题。

**条件敏感性**：方差 = 0.0020（仍然很小）

**闭环仿真**：
| 指标 | 值 |
|------|-----|
| 靠近 Sink 率 | **76%** |
| 平均距离变化 | -12 px（靠近） |
| 初始距离 | 114.6 ± 120.8 |
| 最终距离 | 102.6 ± 121.4 |

**结论**：
1. ✅ 单步预测有改善 (cos_sim 0.26 > 0)
2. ✅ 闭环仿真 76% 靠近率
3. ⚠️ 条件响应仍弱 (方差 0.002)
4. ⚠️ x 方向响应异常，需排查坐标约定

### 10.8 待诊断问题

- [x] x 方向 nav 响应为何反向？→ **已诊断：见 10.9**
- [ ] 条件响应方差为何仍然很小？可能需要更大的 obs_scale 或更长训练
- [ ] 是否需要使用 CFG 推理？(guidance_scale > 1.0)

### 10.9 方向响应不均衡问题 (2025-12-05 诊断)

**现象**：模型对不同导航方向响应不一致
| 方向 | cos mean | >0 比例 | 结论 |
|------|----------|---------|------|
| y+ | 0.40 | 75% | 有效 |
| x+ | 0.18 | 70% | 弱 |
| y- | -0.04 | 45% | 无效 |
| x- | 0.10 | 55% | 近乎随机 |

**根因分析**：

导航场数据本身 **y 方向主导**：
```
nav_y > 0: 33.1%  |  nav_y < 0: 35.9%  → 69% 样本在 y 方向
nav_x > 0: 16.6%  |  nav_x < 0: 14.4%  → 31% 样本在 x 方向
```

这是由 **sink 分布和道路结构** 决定的，不是 bug。

**结论**：
1. 模型正确地学习了数据分布
2. y 方向样本多 → 模型对 y 方向敏感
3. x 方向样本少 → 模型对 x 方向弱

**潜在改进方向**：
- [ ] 数据增强：对样本做旋转/翻转增强
- [ ] 方向重采样：平衡不同方向的样本数量
- [ ] 更长训练：让模型更充分学习稀少方向

### 10.10 CFG 推理效果验证 (2025-12-05)

**测试结果**（20 个位置 × 4 方向）：

| guidance_scale | y+ cos | x+ cos | y- cos | x- cos | 备注 |
|----------------|--------|--------|--------|--------|------|
| 1.0 (无 CFG) | 0.30 | 0.34 | 0.09 | -0.16 | 基线 |
| 2.0 | **0.74** | 0.43 | 0.04 | 0.32 | 改善 |
| **3.0** | **0.80** | **0.68** | 0.00 | 0.29 | **推荐** |
| 5.0 | 0.89 | 0.59 | 0.07 | 0.44 | 过度 |

**结论**：
1. ✅ CFG 显著提升条件响应，尤其是主导方向 (y+, x+)
2. ✅ **推荐使用 guidance_scale=3.0**
3. ⚠️ y- 方向仍然弱，这是数据分布问题

**推理代码**：
```python
# 使用 CFG 推理
samples = scheduler.sample_cfg(
    model=model,
    shape=(B, future, 2),
    condition=obs_flat,
    device=device,
    guidance_scale=3.0,  # 推荐值
)
```

### 10.11 完整评估对比 (2025-12-05)

| 指标 | 无 CFG (gs=1.0) | CFG (gs=3.0) | 提升 |
|------|----------------|--------------|------|
| **Pred vs GT cos_sim** | 0.259 | **0.542** | +109% |
| Pred vs GT >0 率 | 65.7% | **80.4%** | +14.7pp |
| **条件响应方差** | 0.0020 | **0.0940** | +47x |
| **靠近 Sink 率** | 76% | **98%** | +22pp |
| **平均距离变化** | -12 px | **-32 px** | +167% |
| MSE (first frame) | 0.878 | **0.533** | -39% |

**条件响应对比**：
| 方向 | 无 CFG cos | CFG cos |
|------|-----------|---------|
| y+ | 0.75 | **0.97** |
| x+ | -0.95 | **1.00** |
| y- | -0.42 | **0.86** |
| x- | 0.99 | **1.00** |

**最终结论**：
1. ✅ **AdaLN + CFG 组合效果优秀**
2. ✅ **推荐使用 guidance_scale=3.0 进行推理**
3. ✅ 98% 靠近率说明闭环仿真表现良好
4. ⚠️ 推理速度会变慢约 2x (因为 CFG 需要 2 次前向)

### 10.12 模型版本演进汇总 (2025-12-05)

| 版本 | 日期 | 架构 | 关键改动 | 效果 |
|------|------|------|----------|------|
| **v1** | 12月初 | 简单加法 | `h = h + cond_proj(cond)` | cos_sim ≈ 0，condition 被忽略 |
| **v2** | 12/3 | FiLM | `h = gamma * h + beta`，gamma/beta 标准初始化 | 更差，gamma→0 特征被杀死 |
| **v3** | 12/4 | AdaLN | gamma bias=1, obs_scale=2.0 | cos_sim=0.26 (无CFG) |
| **v3+CFG** | 12/5 | AdaLN + CFG推理 | guidance_scale=3.0 | **cos_sim=0.54, 到达率98%** |

**当前最佳: v3 + CFG (best.pt, Epoch 86)**

**各版本详细对比**：

| 指标 | v1 | v2 (FiLM) | v3 (AdaLN) | v3+CFG |
|------|-----|-----------|------------|--------|
| Pred vs GT cos_sim | ~0 | ~0 | 0.26 | **0.54** |
| 条件响应方差 | 0.002 | <0.002 | 0.002 | **0.094** |
| 靠近 Sink 率 | - | - | 76% | **98%** |
| Loss | ~0.30 | ~0.30 | 0.28 | 0.28 |

**关键教训**：
1. 架构改动 (v1→v2→v3) 效果有限
2. **CFG 是关键**：同一模型，CFG 使效果翻倍
3. FiLM 失败是因为初始化问题，不是架构本身

**当前模型文件**：
- `best.pt`: AdaLN 架构, Epoch 86, Loss 0.28, obs_scale=1.77 (训练后)
- 推理时必须用 `guidance_scale=3.0`

### 10.13 Phase 3 vs Phase 4 对比注意事项 (2025-12-05)

**问题**：直接比较 Phase 3 和 Phase 4 轨迹会有偏差

**原因**：
- Phase 3 轨迹包含多次 respawn（agent 到达 sink 后重新出生）
- 很多 agent 的起点就在 sink 附近 (dist=0)
- 导致 `distance_change` 为正（respawn 到远处）

**解决方案**：
对 Phase 3 轨迹，提取**单次 trip**（从 respawn 到下一次 respawn）：
```python
# 找到 respawn 点
respawn_times = np.where(np.diff(dest_seq) != 0)[0] + 1
# 选择起点距离远的段
for start, end in zip(segment_starts, segment_ends):
    start_dist = dist_field[pos[start]]
    if start_dist > threshold:
        use this segment
```

### 10.14 Phase 4 超越 Phase 3 的路径分析 (2025-12-05)

**核心问题**：Phase 4 学习 Phase 3 数据，上限被 Phase 3 限制

**要超越，需要引入 Phase 3 没有的信息**：

| 路径 | 方法 | 可行性 |
|------|------|--------|
| 路径 1 | 引入视觉先验 (CLIP) | 复杂，需要图像数据 |
| **路径 2** | **OD Flow 作为额外监督** | ✅ 已有数据 |
| 路径 3 | Energy-Based 直接学 OD 约束 | 理论可行 |
| 路径 4 | Diffusion + RL 微调 | 需要设计 reward |

**推荐方案：OD-Aware Diffusion Policy**

现有 OD 数据分析：
- `sink_od_prob.csv`: 35×35 的 OD 概率矩阵
- 平均熵 2.88 < 最大熵 3.56 → **Destination 选择不是均匀分布**
- 这是可以学习的真实知识！

**具体实施思路**：
```
当前: Phase 4 只学 (obs, nav) → velocity
改进: Phase 4 学 (obs, nav, destination_id) → velocity

推理时:
1. 根据 origin 位置确定 origin_sink
2. 用真实 OD prob 采样 destination_sink  
3. 生成去往该 destination 的轨迹
```

**泛化能力的关键**：
- 不是学习特定轨迹，而是学习 **"如何根据目的地选择路径"**
- OD 分布是真实数据，不是仿真假象
- 模型需要学到：同一起点，不同目的地 → 不同的行为模式

### 10.15 待实现改进

- [ ] 给 Phase 4 模型增加 destination embedding 作为 condition
- [ ] 用真实 OD prob 采样 destination（替代 Phase 3 的随机采样）
- [ ] 设计评估指标：生成轨迹的 OD 分布与真实 OD 的 KL 散度

### 10.16 Phase 3 评估方法错误 (2025-12-05 诊断)

**问题现象**：
之前的对比图显示 Phase 3 的 approaching rate = 0%，这显然不合理。

**根本原因**：
评估脚本使用了 `distance_field.npy`（到**最近** sink 的距离），而非到**目标** sink 的距离。

**Phase 3 仿真机制回顾**：
1. Agent 从 sink A 出发，目标是 sink B
2. 在 A 点，到最近 sink 的距离 = 0（就是 A 本身）
3. 移动过程中，距离先增大（离开 A），后减小（靠近 B）
4. 但如果 A 和 B 距离近，全程距离都很小

**正确评估方法**：
使用每个 sink 的独立距离场 `nav_fields/nav_field_XXX.npz['distance_field']`

```python
# 正确：到目标 sink 的距离
target_sink = dest[t, agent]
dist_to_target = sink_dist_fields[target_sink][y, x]

# 错误：到最近 sink 的距离
dist_to_nearest = distance_field[y, x]  # 这是之前用的
```

**Phase 3 真实表现**（使用正确评估，100 个筛选样本，300 步）：
| 指标 | 值 |
|------|-----|
| 起点到目标距离 | 401.6 px |
| 终点到目标距离 | 157.3 ± 227.0 px |
| **距离变化** | **-244.4 ± 81.2 px**（靠近） |
| **靠近目标率** | **100%** |
| **到达率** | **44%** |
| 速度 | 1.13 ± 0.13 px/step |
| 平滑度 | 25.9°/step |

**筛选条件**：
- 起点到目标距离 > 50 px
- Trip 长度 >= 100 步

**结论**：
Phase 3 仿真本身是正确的，问题出在评估方法。

### 10.17 公平对比评估脚本 (2025-12-05)

**脚本**: `scripts/compare_phase3_phase4_fair.py`

**关键修正**:
1. Phase 4 推理时使用**目标 sink 的导航场**（而非全局 nav_baseline）
2. 评估时使用**目标 sink 的距离场**
3. 使用筛选后的有效样本（起点距离 > 50）

**运行方式**:
```bash
# 先生成有效样本
python scripts/filter_eval_samples.py

# 再运行公平对比（需要 GPU）
python scripts/compare_phase3_phase4_fair.py --n_samples 100 --guidance 3.0
```

**输出**:
- `data/output/phase4_validation/phase3_vs_phase4_fair_comparison.png`

### 10.18 公平对比结果分析 (2025-12-05)

**实验设置**：
- 100 个筛选样本（起点距离 > 50 px）
- 相同起点、相同目标 sink
- Phase 4 使用 CFG=3.0
- 最大 300 步

**结果**：
| 指标 | Phase 3 | Phase 4 |
|------|---------|---------|
| 起点距离 | 401.6 | 401.6 |
| 终点距离 | 157.3 ± 227.0 | 339.3 ± 269.6 |
| **距离变化** | **-244.4** | -62.3 |
| 靠近率 | 100% | 99% |
| **到达率** | **44%** | 1% |
| 速度 | 1.13 | 0.92 |
| 平滑度 | 26°/step | 60°/step |

**效率分析**：
| 指标 | Phase 3 | Phase 4 |
|------|---------|---------|
| 总路程 | 339 px | 276 px |
| 有效前进 | 244 px | 62 px |
| **效率** | **72%** | **22.5%** |

**问题诊断**：
Phase 4 虽然 99% 在靠近目标，但效率极低（22.5%），77.5% 的移动是无效的：
1. 方向抖动大（60°/step vs 26°/step）
2. 速度偏慢（0.92 vs 1.13）
3. 可能在目标附近震荡

**结论**：
- Phase 4 学到了"方向大致正确"（99% approaching）
- 但没有学到"高效前进"（只有 22.5% 效率）
- 当前 Phase 4 **显著弱于 Phase 3**

### 10.19 推理代码 Bug：vel_hist 记录错误 (2025-12-05 发现)

**问题**：
推理代码中 `vel_hist` 记录的是**预测的 action**，而非**实际位移**。

当道路约束阻止移动时：
- 位置不变 (`pos` 不更新)
- 但 `vel_hist.append(vel)` 记录了预测的 action
- 下一次推理时，obs 中 pos 和 vel 不一致

**影响**：
- 训练数据中 `vel = pos[t+1] - pos[t]`，位置相同时 vel = 0
- 推理时 vel 可能非零但位置相同
- 导致 obs 分布偏离训练分布，模型行为异常

**修复**：
```python
# 之前（错误）：
vel_hist.append(vel.copy())  # vel = actions[j]

# 修复后：
old_pos = pos.copy()
# ... 更新 pos ...
actual_vel = pos - old_pos  # 实际位移
vel_hist.append(actual_vel)
```

**修复文件**：
- `scripts/compare_phase3_phase4_fair.py`
- `scripts/analyze_phase4_jitter.py`

---

*最后更新: 2025-12-05*