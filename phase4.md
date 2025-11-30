这是一份为你量身定制的 `DEVELOPMENT_PHASE4.md` 文档。

这份文档标志着你的项目正式进入 **"Generative AI"** 的核心领域。我们不再是用物理方程（PDE）或者简单的回归网络来导航，而是训练一个能够\*\*像人一样思考（基于概率分布决策）\*\*的智能体。

-----

### 📥 建议行动

1.  下载此文档。
2.  将其放入项目根目录。
3.  当你完成 Phase 3 的数据生成后，将此文档发给 Codex 启动 Phase 4 开发。

-----

````markdown
# Phase 4: Generative Motion Modeling (Diffusion Policy)

## 1. 项目目标 (Objective)
**当前阶段**: Phase 4 - 生成式行为克隆 (Diffusion Policy)

**核心理念**:
- 以 Phase 3 合成的轨迹数据（`trajectories.h5`）为燃料，学习条件分布 $p_\theta(A | O)$。
- 通过扩散去噪网络 (UNet1D) 预测未来动作分布，实现多模态、概率化的导航策略。

**输入**: Phase 3 产出的 `trajectories.h5` (大规模合成轨迹)。
**输出**: 预训练好的扩散策略模型，可在新初始位置下生成合理的动作序列。

---

## 2. 核心模块与代码结构

```
src/phase4/
├── config.py                # 数据路径、窗口长度等超参
├── data/
│   ├── dataset.py           # HDF5 滑窗 Dataset，obs=2帧(pos,vel)，action=未来8帧vel
│   └── normalizer.py        # MinMax/Z-Score 归一化
├── model/
│   └── unet1d.py            # 条件 1D UNet (timestep + global cond)
├── diffusion/
│   └── scheduler.py         # DDPM/可选 DDIM 调度
├── train.py                 # 训练脚本，EMA 平滑 + MSE 噪声预测
└── inference.py             # 推理/MPC：预测未来8步，执行1步
```

## 3. 训练与推理

### 训练
```
cd e:\newdesktop\HKUST\GeoExplicit_SFM\v2
python src/phase4/train.py --epochs 100 --batch_size 256
```
关键点：
- 使用 `TrajectorySlidingWindow` 读取 `(obs, action)`；obs 包含过去2帧位置+速度，action 为未来8帧速度。
- `ActionNormalizer` 默认 MinMax 到 [-1,1]。
- 调度器 `DDPMScheduler` 管理扩散/去噪；UNet1D 条件输入包含 timestep embedding + 全局 cond。
- EMA 平滑权重。

### 推理
```
python src/phase4/inference.py --num_agents 20 --max_steps 500
```
关键点：
- MPC 风格：预测 8 步，执行 1 步再滚动。
- 支持 DDPM/可选 DDIM 采样加速。

## 4. 设计要点
- Observation: 过去 2 帧的 (pos, vel) → obs_dim=4；Action: 未来 8 帧 vel。
- Conditional UNet1D：timestep sinusoidal embedding + 全局条件（观测投影）。
- 归一化：MinMax 到 [-1,1]（`normalizer.py`）。
- 调度：`scheduler.py` 提供 beta schedule、前向/反向公式。
2.  加入高斯噪声 $\epsilon \sim \mathcal{N}(0, I)$ 得到 $A_k$ (第 $k$ 步加噪)。
3.  训练神经网络预测噪声：$\epsilon_\theta(A_k, k, O) \approx \epsilon$。

---

## 3. 技术栈 (Tech Stack)
* **Framework**: PyTorch
* **Library**:
    * `diffusers` (HuggingFace): 提供 Scheduler (Ddim/Ddpm) 的标准实现。
    * `h5py`: 高效读取 Phase 3 的大数据文件。
    * `wandb`: 实验记录与 Loss 曲线监控。

---

## 4. 项目文件结构 (Directory Structure)

```text
src/phase4/
├── __init__.py
├── config.py             # 训练超参 (Batch, LR, Diffusion Steps)
├── data/
│   ├── __init__.py
│   ├── dataset.py        # [关键] HDF5 懒加载, 滑动窗口切片 (Horizon)
│   └── normalizer.py     # 数据的归一化与反归一化 (MinMax/Z-Score)
├── model/
│   ├── __init__.py
│   ├── unet1d.py         # [核心] 1D-UNet 或 Transformer 用于处理时序动作
│   └── encoder.py        # 将环境信息 (Mask/Field) 编码为 Embedding
├── diffusion/
│   ├── __init__.py
│   └── scheduler.py      # 管理噪声添加与采样 (DDPM/DDIM)
├── train.py              # 训练主循环
└── inference.py          # 闭环测试：用模型驱动 Agent 跑完全程
````

-----

## 5\. 模块功能详述 (Module Specifications)

### A. `src/phase4/data/dataset.py`

**功能**：高效的数据管道。

  * **Context Length (Obs Horizon)**: 过去 $T_{obs}$ 步的状态（例如过去 0.5秒）。
  * **Prediction Horizon**: 预测未来 $T_{pred}$ 步的动作（例如未来 1.0秒）。
  * **逻辑**：
      * 打开 HDF5 文件（只读模式，不读入内存）。
      * 实现 `__getitem__`: 随机选取一条轨迹的切片。
      * **Input**: `obs = {pos, vel, local_map_crop}`
      * **Target**: `action = {future_velocities}`

### B. `src/phase4/model/unet1d.py`

**功能**：去噪网络。

  * 不同于 Phase 2 处理图像的 2D-UNet，这里我们需要处理**时间序列**。
  * 推荐使用 **1D Conditional UNet** (类似 Diffusion Policy 原文架构)。
  * **Input**: Noisy Action Sequence (Batch, T, Dim) + Condition Embedding.
  * **Output**: Predicted Noise (Batch, T, Dim).

### C. `src/phase4/train.py`

**功能**：训练循环。

  * 标准 Diffusion 训练流程。
  * **关键点**：一定要使用 `EMA (Exponential Moving Average)` 来平滑模型权重，这对于 Diffusion 的生成质量至关重要。

### D. `src/phase4/inference.py` (闭环验证)

**功能**：这也是论文的 Figure 生成器。

  * 初始化一批 Agent 在随机位置。
  * **Simulation Loop**:
    1.  获取当前状态 $O_t$。
    2.  **Denoising**: 从随机噪声开始，用训练好的模型迭代 $K$ 步，生成动作序列 $A_{pred}$。
    3.  **Execution**: 执行 $A_{pred}$ 的第一步（Model Predictive Control, MPC 模式）。
    4.  物理环境更新位置。
    5.  重复。

-----

## 6\. 开发路线图 (Roadmap)

请 Codex 按此顺序执行：

#### Step 1: 数据加载器 (The Feeder)

  * [ ] 实现 `dataset.py`。
  * *任务*: 能从 Phase 3 的 HDF5 中读取 Batch。
  * *验证*: 打印出一个 Batch 的 Shape，确认归一化是否正确 (Mean≈0, Std≈1)。

#### Step 2: 模型构建 (The Brain)

  * [ ] 实现 `model/unet1d.py`。
  * *任务*: 定义一个接受 (B, T, D) 输入的 Conditional 1D UNet。
  * *验证*: 随机输入 Tensor，确认输出 Shape 不变。

#### Step 3: 训练循环 (The Training)

  * [ ] 实现 `train.py`。
  * *任务*: 跑通 DDPM 训练流程。
  * *目标*: Loss 应该能降到一个稳定的数值（代表成功学会了分布）。

#### Step 4: 闭环推理 (The Result)

  * [ ] 实现 `inference.py`。
  * *任务*: 加载模型，在一个空白的 `Walkable Mask` 上放一个粒子，看它是否能像 Phase 3 那样“自动”避开墙壁并走向终点。
  * *SOTA 时刻*: 观察粒子是否表现出了比纯物理模型更“平滑”或“拟人”的特征。

-----



```
```
