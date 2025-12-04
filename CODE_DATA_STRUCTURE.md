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

### 2.2 导航场 `data/processed/nav_fields/`

```
目录结构:
nav_fields/
├── nav_fields_index.json     # 索引文件
├── nav_field_000.npz         # sink 0 的导航场
├── nav_field_001.npz         # sink 1 的导航场
└── ...

单个 .npz 文件内容:
├── nav_y: (H, W) float32     # y 方向分量
└── nav_x: (H, W) float32     # x 方向分量

加载后组合为: (2, H, W) 格式，即 [nav_y, nav_x]
```

**加载代码**：
```python
def load_nav_fields(nav_fields_dir):
    """加载所有个体导航场"""
    nav_fields = {}
    index = json.load(open(nav_fields_dir / "nav_fields_index.json"))
    for sink_id in index["sink_ids"]:
        data = np.load(nav_fields_dir / f"nav_field_{sink_id:03d}.npz")
        nav_fields[sink_id] = np.stack([data["nav_y"], data["nav_x"]], axis=0)  # (2, H, W)
    return nav_fields
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

*最后更新: 2025-12-04*

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

4. **根本原因分析**
   - 扩散模型学会了预测"平均噪声"，忽略 condition
   - 这是条件扩散模型训练的经典问题
   - 训练数据分布没问题 (静止样本仅 0.1%，方向分布均匀)

**推荐解决方案:**

1. **Classifier-Free Guidance (CFG)** - 最推荐
   ```python
   # 训练时随机丢弃 condition (如 10% 概率)
   if random.random() < 0.1:
       condition = torch.zeros_like(condition)  # 无条件
   
   # 推理时使用 CFG 公式增强 condition 影响
   eps_uncond = model(x, t, zeros_condition)
   eps_cond = model(x, t, real_condition)
   eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
   # guidance_scale 通常 2.0 ~ 7.5
   ```

2. **增大 condition embedding** (简单尝试)
   - 当前 cond_dim=64 可能太小
   - 尝试 cond_dim=128 或 256

3. **FiLM 条件注入** (架构改进)
   ```python
   # Feature-wise Linear Modulation
   gamma, beta = cond_proj(condition)  # (B, C)
   h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)  # 乘法 + 加法
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

