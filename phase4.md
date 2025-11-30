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
**当前阶段**: Phase 4 - 生成式行为克隆 (Generative Behavioral Cloning)

**核心理念**:
Phase 3 的物理引擎是一个“高精度的规则生成器”。
Phase 4 的目标是训练一个 **Diffusion Policy**，从 Phase 3 生成的数百万条轨迹数据中学习运动规律 ($State \to Action$)。
它不再是机械地执行 $F=ma$，而是学会了**“在当前环境下，应该以什么样的概率分布采取行动”**。这是通往 **Sim-to-Real** 和 **多模态导航** 的关键一步。

**输入**: Phase 3 产出的 `trajectories.h5` (大规模合成轨迹)。
**输出**: 一个预训练好的扩散策略模型 (Diffusion Model)，能够在一个从未见过的初始位置，自主规划出前往 CBD 的合理路径。

---

## 2. 核心数学逻辑 (The Math)

我们将导航问题建模为 **条件扩散过程 (Conditional Diffusion Process)**。

* **Observation ($O$)**: 智能体的状态（位置、速度）+ 环境信息（局部地图特征、目标方向）。
* **Action ($A$)**: 智能体的下一步动作（速度增量 $\Delta v$ 或 加速度）。
* **模型目标**: 学习条件分布 $p_\theta(A | O)$。

训练过程（DDPM）：
1.  从数据集中采样一个真实的动作 $A_0$。
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