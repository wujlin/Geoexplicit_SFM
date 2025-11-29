你的提醒非常及时！这是我在整理文档时的疏忽。

确实，我们之前的战略定调是：**“双轨制 (Dual-Track)”**。

1.  **Baseline (基准线)**：使用传统的、可解释的 **物理数值解法 (PDE Solver)**。这保证了下限，绝对能跑通。
2.  **Innovation (创新线)**：引入 **Neural Network (Score-Based Model)**。这决定了上限，用于捕捉非线性的宏观趋势，并提升论文的 SOTA 属性。

我重新重构了 `DEVELOPMENT_PHASE2.md`，采用了**模块化双轨架构**。

-----

### 使用说明

这份文档明确区分了 `baseline` 和 `innovation` 两个子模块。

  * 你可以让 Codex 先写 Baseline，确保项目能跑。
  * 然后再写 Innovation，去挑战更高的性能或更好的平滑度。
  * 最后有一个 **Comparison** 环节，这直接就是你论文里的 Experiment Section。

-----

````markdown
# Geo-Explicit SFM Simulator - Development Guide (Phase 2)

## 项目概述 (Project Overview)
**当前阶段**: Phase 2 - 扩散增强型场生成 (Diffusion-Enhanced Field Generation)

**战略路线 (Strategy)**: 采用“双轨制”开发。
* **Track A (Baseline)**: **物理数值求解 (Numerical PDE Solver)**。基于几何受限的热传导方程，直接计算稳态密度场及其梯度。保证物理约束的绝对刚性。
* **Track B (Innovation)**: **神经评分场 (Neural Score Field)**。利用 **Score-Based Generative Model (SGM)** 的思想，训练一个轻量级 UNet 来预测导航场 (Score)。旨在捕捉比纯物理扩散更复杂的宏观趋势，并探索“AI for Science”的结合点。

**核心输入**:
* Phase 1 输出的汇坐标 `sinks_phase1.csv`。
* OSM 路网数据。
* WorldPop 人口密度数据 (作为 Neural Network 的额外 Condition)。

---

## 1. 技术栈 (Tech Stack)
* **Common**: `osmnx`, `rasterio`, `numpy`, `matplotlib`
* **For Baseline**: `scipy.ndimage` (卷积/拉普拉斯算子), `scipy.sparse` (大型线性方程组求解)
* **For Innovation**: `pytorch` (构建 UNet 和 Score Matching Loss)

---

## 2. 项目文件结构 (Directory Structure)

结构设计支持双轨并行开发：

```text
geo-sfm-sim/
├── data/
│   ├── processed/
│   │   ├── sinks_phase1.csv         # [Input]
│   │   ├── walkable_mask.npy        # [Common Output] 路网掩膜
│   │   ├── field_baseline.npy       # [Output Track A] 物理求解的场
│   │   └── field_neural.npy         # [Output Track B] 神经网络预测的场
│   └── training/                    # 仅用于 Track B 的训练数据缓存
├── src/
│   ├── phase2/
│   │   ├── __init__.py
│   │   ├── config.py                # 全局参数
│   │   ├── common/                  # 公共模块
│   │   │   ├── geo_rasterizer.py    # 路网栅格化
│   │   │   └── visualizer.py        # 流线对比绘图
│   │   ├── baseline/                # Track A: 物理求解
│   │   │   └── pde_solver.py
│   │   └── innovation/              # Track B: 神经网络
│   │       ├── dataset.py           # 构造 (Geometry, Density) 对
│   │       ├── network.py           # Simple UNet / ScoreNet
│   │       └── trainer.py           # 训练循环
├── main_phase2_baseline.py          # 执行 Track A
├── main_phase2_innovation.py        # 执行 Track B
└── DEVELOPMENT_PHASE2.md            # 本文档
````

-----

## 3\. 模块功能详述 (Module Specifications)

### A. 公共模块 (Common)

#### `src/phase2/common/geo_rasterizer.py`

**功能**：为两条路线提供统一的“物理容器”。
**逻辑**：

1.  下载 BBox 内的 OSM 骨干路网。
2.  **双层栅格化**：
      * Layer 1: `walkable_mask` (0/1) - 用于 Baseline 的扩散系数 $D(x)$。
      * Layer 2: `target_density_map` (Float) - 结合 Phase 1 的 Sinks (高斯核) 和 WorldPop 数据。这将作为 Baseline 的源项，以及 Innovation 网络的 Input/Target 基础。

-----

### B. Track A: Baseline (Physics-Based)

#### `src/phase2/baseline/pde_solver.py`

**原理**：求解稳态扩散方程（泊松方程）。
$$\nabla \cdot (D(\mathbf{x}) \nabla \rho) = -\rho_{target}(\mathbf{x})$$
或者使用迭代热传导：$\rho_{t+1} = \rho_t + \alpha \nabla \cdot (D \nabla \rho_t)$。

**开发任务**：

1.  **迭代求解器**: 实现基于 `scipy.ndimage.convolve` 的有限差分迭代。
      * *关键点*: 扩散系数 $D(\mathbf{x})$ 在路网处为 1.0，非路网处为 $10^{-4}$。这保证了场主要沿路网传播。
2.  **梯度计算**:
      * 计算 $\rho_{final}$ 的梯度 $\nabla \rho$。
      * 转换为 Score: $\mathbf{v} = \nabla \log \rho = \frac{\nabla \rho}{\rho + \epsilon}$。
      * 归一化模长。

-----

### C. Track B: Innovation (Neural Network)

#### `src/phase2/innovation/network.py`

**原理**：Score-Based Generative Modeling。
我们不生成图片，我们训练网络预测 **梯度场 (Score Field)**。

  * **Input**: `[Walkable_Mask (1ch), Target_Density_Map (1ch)]` (2通道张量，包含几何与宏观趋势)。
  * **Output**: `[Score_Field_X, Score_Field_Y]` (2通道张量，即导航向量)。
  * **Model**: 一个轻量级的 **UNet** 或 **ResNet**。

#### `src/phase2/innovation/trainer.py`

**训练目标 (Self-Supervised / Physics-Informed)**：
由于没有 Ground Truth 轨迹，我们采用 **Denoising Score Matching (DSM)** 的变体，或者 **Physics-Informed Loss (PINN)**。

**推荐方案 (Score Matching with Density Construction)**:

1.  **构造训练样本**:
      * 从 `target_density_map` $\rho(\mathbf{x})$ 中采样点 $\mathbf{x} \sim \rho(\mathbf{x})$。
      * 这些点主要分布在 CBD 和主干道上。
2.  **扰动与去噪 (Denoising Objective)**:
      * 给采样点 $\mathbf{x}$ 加噪声 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \mathbf{z}$。
      * 让网络预测噪声方向：$Loss = || s_\theta(\tilde{\mathbf{x}}) \cdot \sigma + \mathbf{z} ||^2$。
      * *物理含义*: 网络学会了“如果我偏离了主干道/CBD（被加了噪声），我该往哪个方向走才能回去”。
      * *创新点*: 这里的 $\rho(\mathbf{x})$ 融合了 WorldPop 和 OSM，所以网络学出的是一个\*\*“人口-几何混合势场”\*\*。

-----

## 4\. 开发路线图 (Roadmap)

请 Codex 按此顺序执行开发：

#### Step 1: 基础设施 (Common)

  * [ ] 实现 `config.py`。
  * [ ] 实现 `geo_rasterizer.py`。
  * *验收*: 运行脚本，输出 `walkable_mask.npy` 和 `target_density.npy`，检查路网是否清晰，CBD 处密度是否高。

#### Step 2: 实现 Baseline

  * [ ] 编写 `src/phase2/baseline/pde_solver.py`。
  * [ ] 编写 `main_phase2_baseline.py`。
  * *验收*: 生成 `field_baseline.npy` 和流线图。流线应沿着路网汇聚到 Sink。

#### Step 3: 实现 Innovation

  * [ ] 搭建 `network.py` (Simple UNet)。
  * [ ] 编写 `trainer.py` (实现 Score Matching Loss)。
  * [ ] 编写 `main_phase2_innovation.py` (包含训练循环和推断)。
  * *验收*: 训练 loss 下降；输出 `field_neural.npy`。

#### Step 4: 对比分析 (Comparison)

  * [ ] 编写 `compare_fields.py`。
  * *任务*: 将 Baseline 和 Innovation 的流线图并排画在一起。
  * *预期差异*: Baseline 的场可能比较生硬（纯几何）；Innovation 的场可能更平滑，且对 WorldPop 的密度分布有更强的“泛化响应”（比如在没有路网的空地，也能预测出大致的宏观方向）。

-----

## 5\. 给 Codex 的 Prompt 建议

**启动 Phase 2 开发时，请发送以下指令**：

> "这是 Phase 2 的开发文档。我们将采用双轨制：Baseline 使用 PDE 求解，Innovation 使用 Neural Network。
> 请先帮我完成 **Step 1: Common Infrastructure**。
> 我需要 `geo_rasterizer.py`，它需要从 OSM 下载路网，并结合 Phase 1 的 Sinks 生成路网掩膜和目标密度图。"

**完成 Baseline 后，发送**：

> "现在开始 Track A: Baseline。请实现 `pde_solver.py`，使用有限差分法在 `walkable_mask` 上对 `target_density` 进行平滑扩散，并计算梯度场。"

**完成 Innovation 时，发送**：

> "现在开始 Track B: Innovation。我们要训练一个 UNet 来学习 Score Field。
> 请实现 `network.py` 和 `trainer.py`。
> 训练逻辑是：从 `target_density` 中采样点位置，加高斯噪声，训练网络预测噪声方向 (Denoising Score Matching)。"

-----

```
```