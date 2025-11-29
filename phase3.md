这是为你准备的 `DEVELOPMENT_PHASE3.md` 文档。

这一阶段是将前两个阶段的静态成果“激活”为动态生命的关键步骤。它将 Phase 1 的**硬约束**和 Phase 2 的**软引导**融合进微观粒子的**朗之万动力学**中，完成整个 Geo-Explicit SFM 的闭环。

-----

````markdown
# Geo-Explicit SFM Simulator - Development Guide (Phase 3)

## 项目概述 (Project Overview)
**当前阶段**: Phase 3 - 微观动力学演化与分析 (Micro-Dynamics & Analytics)

**前置依赖**:
* Phase 1 输出: `walkable_mask.npy` (几何约束), `sinks_phase1.csv` (目标汇).
* Phase 2 输出: `field_baseline.npy` 或 `field_neural.npy` (宏观导航场).

**Phase 3 核心目标**:
构建高性能的 **Agent-Based Model (ABM)** 引擎。
将宏观的 Census OD 通量转化为微观的粒子注入事件，利用 **朗之万方程 (Langevin Equation)** 驱动粒子在 Phase 2 生成的场中运动，同时计算微观层面的社会力 (Social Force) 避让。
最终产出：城市的动态拥堵热力图和结构性阻力分析。

---

## 1. 技术栈 (Tech Stack)
* **Language**: Python 3.9+
* **Compute Engine**:
    * `numba` (推荐) 或 `taichi`: 用于加速粒子循环和邻居搜索，避免 Python 原生循环的低效。**这是必须的**，因为我们要跑数万个粒子。
    * `scipy.interpolate`: 用于在连续坐标 $(x,y)$ 处读取离散网格场 $(i,j)$ 的数值。
    * `pandas`: 管理 Agent 的状态数据。
    * `matplotlib` / `seaborn`: 结果可视化与图表绘制。

---

## 2. 项目文件结构 (Directory Structure)

```text
geo-sfm-sim/
├── data/
│   ├── processed/
│   │   ├── field_baseline.npy       # [Input]
│   │   ├── field_neural.npy         # [Input]
│   │   └── walkable_mask.npy        # [Input]
│   └── output/
│       ├── trajectories.h5          # [Output] 粒子轨迹数据 (HDF5格式)
│       └── metrics/                 # [Output] 分析结果
│           ├── congestion_map.png
│           └── impedance_analysis.csv
├── src/
│   ├── phase3/
│   │   ├── __init__.py
│   │   ├── config.py                # 物理参数 (dt, mass, tau, noise)
│   │   ├── environment.py           # 负责读取 Mask/Field 并提供插值接口
│   │   ├── spawner.py               # [关键] 将 Census Flux 转化为生成事件
│   │   ├── physics_engine.py        # [核心] Numba 加速的 Verlet 积分器
│   │   └── analytics.py             # 密度计算与阻力推断
├── main_phase3.py                   # 执行入口
└── DEVELOPMENT_PHASE3.md            # 本文档
````

-----

## 3\. 物理模型设计 (Physics Logic)

粒子的运动遵循 **欠阻尼朗之万方程 (Underdamped Langevin Dynamics)**：

$$
m \dot{\mathbf{v}}_i = \mathbf{F}_{macro} + \mathbf{F}_{geo} + \mathbf{F}_{soc} + \mathbf{F}_{noise} - \gamma \mathbf{v}_i
$$

### 力项分解：

1.  **$\mathbf{F}_{macro}$ (宏观导航)**:
      * 来源：Phase 2 的 `field_*.npy`。
      * 逻辑：`force = (field_velocity(pos) - current_velocity) / relaxation_time`。
      * *作用*：让粒子顺应宏观趋势去上班。
2.  **$\mathbf{F}_{geo}$ (几何避障)**:
      * 来源：Phase 1 的 `walkable_mask`。
      * 逻辑：计算 Mask 的 SDF 梯度。如果粒子试图进入墙壁，给予强反向斥力。
3.  **$\mathbf{F}_{soc}$ (微观避让)**:
      * 来源：粒子 $i$ 与邻居 $j$ 的距离。
      * 逻辑：$F_{ij} = A \exp((r_{ij} - d_{ij}) / B)$。
      * *作用*：模拟拥挤、排队和相变。
4.  **$\mathbf{F}_{noise}$ (热涨落)**:
      * 来源：高斯白噪声。
      * *作用*：模拟个体行为的不确定性。

-----

## 4\. 模块功能详述 (Module Specifications)

### A. `src/phase3/config.py`

**功能**：物理常数配置。

  * `DT`: **0.1** (时间步长, 秒)。
  * `MAX_AGENTS`: **10000** (最大同屏人数限制)。
  * `NEIGHBOR_RADIUS`: **2.0** (米, 感知半径)。
  * `FIELD_TYPE`: `'baseline'` 或 `'neural'` (切换开关)。

### B. `src/phase3/environment.py`

**功能**：静态环境管理器。
**核心逻辑**：

1.  加载 `.npy` 文件。
2.  **SDF 计算**: 在初始化时，使用 `scipy.ndimage.distance_transform_edt` 将 `walkable_mask` (0/1) 转换为 `wall_sdf` (距离场)。
3.  **插值器 (Interpolator)**: 提供 `get_field_vector(x, y)` 和 `get_wall_distance(x, y)` 方法。
      * *性能提示*: 尽量使用最近邻插值或双线性插值，为了速度可以牺牲一点精度。

### C. `src/phase3/spawner.py` (数据驱动核心)

**功能**：时空降尺度注入器。
**输入**: Census OD Data (`mi_od_main_JT00_2021.csv`).
**逻辑**:

1.  **Flux to Rate**: 将 OD 表中的 `flow` (人/年) 转换为 `spawn_rate` (人/秒)。
      * 公式: `rate = flow * PEAK_FACTOR / 3600`。
2.  **Queue Generation**:
      * 预生成一个事件队列: `[(time_0, pos_A), (time_1, pos_B), ...]`。
      * `pos` 的生成采用 Rejection Sampling：在 Origin Tract 内随机撒点，只有落在 `walkable_mask` 且人口密度高的地方才保留。
3.  **Step**: 每一帧检查队列，如果 `current_time > event_time`，则从队列取出放入仿真器。

### D. `src/phase3/physics_engine.py` (高性能核心)

**功能**：基于 Numba/Taichi 的积分循环。
**数据结构**: 使用 Numpy Arrays 存储状态 `POS[N, 2]`, `VEL[N, 2]`, `ACTIVE[N]`。
**核心函数 (JIT 编译)**:

1.  `update(pos, vel, field_map, sdf_map)`:
      * **Step 1: 查表**。根据 `pos` 获取宏观力 $F_{macro}$ 和 墙壁力 $F_{geo}$。
      * **Step 2: 邻居搜索**。使用简单的 Grid Hash (网格法) 寻找 2米内的邻居，累加 $F_{soc}$。
      * **Step 3: 积分**。应用 Velocity Verlet:
        ```python
        acc = (F_total - gamma * vel) / mass
        vel += acc * dt
        pos += vel * dt
        ```
      * **Step 4: 边界检查**。如果 `pos` 到达 Sink 附近，标记 `ACTIVE[i] = False` (移除)。

### E. `src/phase3/analytics.py`

**功能**：从微观轨迹中提取宏观洞察。
**核心逻辑**:

1.  **密度图 (Density Map)**: 统计网格内的平均粒子数 $\langle N \rangle$。
2.  **速度图 (Velocity Map)**: 统计网格内的平均速度 $\langle |v| \rangle$。
3.  **阻力推断 (Impedance Inference)**:
      * 计算公式: $Z(x) = \frac{V_{free}}{\langle |v(x)| \rangle + \epsilon}$。
      * *物理含义*: 如果 $Z \approx 1$，畅通；如果 $Z > 5$，说明发生了结构性拥堵。

-----

## 5\. 开发步骤 (Step-by-Step Instructions)

请 Codex 按照以下顺序执行：

#### Step 1: 环境搭建

  * [ ] 实现 `environment.py`。
  * *任务*: 加载 Phase 2 的 Field 和 Phase 1 的 Mask，计算 SDF。编写一个测试脚本，输入坐标 $(100, 100)$，打印出该处的场向量和离墙距离。

#### Step 2: 生成器逻辑

  * [ ] 实现 `spawner.py`。
  * *任务*: 读取 Census CSV，实现“年度流量 -\> 秒级生成率”的转换逻辑。打印出“早高峰一小时内将生成 X 个粒子”。

#### Step 3: 物理引擎 (The Hard Part)

  * [ ] 实现 `physics_engine.py`。
  * *任务*: 先用纯 Python 写通逻辑，然后加上 `@numba.jit(nopython=True)` 装饰器进行加速。
  * *关键*: 实现简单的 Grid-based 邻居搜索，不要暴力 $O(N^2)$。

#### Step 4: 主循环与集成

  * [ ] 编写 `main_phase3.py`。
  * *逻辑*:
    ```python
    env = Environment(...)
    spawner = Spawner(...)
    engine = PhysicsEngine(...)

    for t in range(TIMESTEPS):
        new_agents = spawner.step(t)
        engine.add_agents(new_agents)
        engine.update(env)
        if t % 100 == 0:
            analytics.snapshot(engine) # 保存快照
    ```

#### Step 5: 分析与可视化

  * [ ] 运行模拟。
  * [ ] 使用 `analytics.py` 生成 `congestion_heatmap.png`。
  * *验证*: 对比 `FIELD_TYPE='baseline'` 和 `FIELD_TYPE='neural'` 下，拥堵点的分布是否有区别？

-----

## 6\. 给 Codex 的 Prompt 建议

**启动 Phase 3 开发时，请发送**：

> "这是 Phase 3 的开发文档。我们要构建一个基于 Numba 加速的 SFM 模拟引擎。
> 请首先帮我实现 `environment.py`，它需要加载之前生成的 `.npy` 文件，并提供高效的插值接口供物理引擎调用。"

**开发物理引擎时，发送**：

> "现在实现核心物理引擎 `physics_engine.py`。
> 请使用 Numba 的 `@jit` 来加速。
> 包含三个力的计算：
>
> 1.  Macro Force (从 Environment 插值)。
> 2.  Geo Force (基于 SDF 梯度的斥力)。
> 3.  Social Force (基于网格搜索的邻居斥力)。
>     最后用 Velocity Verlet 更新位置。"

-----

```
```