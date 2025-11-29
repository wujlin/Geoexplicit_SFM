这是一个为你精心撰写的 `DEVELOPMENT_PHASE1.md` 文档。它是专门为 AI 辅助编程（如 Codex CLI / Cursor / GitHub Copilot）设计的，结构清晰，指令明确。

你可以下载这个文件，然后直接发给 Codex，对它说：“**请根据这个文档的架构，帮我搭建项目并实现 Phase 1 的功能。**”

-----

### 使用建议

1.  **创建文件夹**：先在本地建一个项目文件夹（如 `geo-sfm-sim`）。
2.  **放入数据**：在 `data/raw/` 下放入你下载好的 Census LODES 数据（csv）和 Shapefiles。
3.  **运行 Codex**：将此文档内容提供给 Codex，它会知道该创建哪些文件、写什么函数。

-----

````markdown
# Geo-Explicit SFM Simulator - Development Guide (Phase 1)

## 项目概述
本项目旨在构建一个**数据驱动 (Data-Driven)** 且 **地理显式 (Geo-Explicit)** 的城市人流模拟器。
项目分为多个阶段。本文档仅关注 **Phase 1：数据清洗与核心汇识别 (Data Preparation & Sink Identification)**。

**Phase 1 核心目标**：
从海量的宏观 Census OD (Origin-Destination) 数据中，利用统计学方法（帕累托法则 + DBSCAN 聚类），自动识别出城市中的 $K$ 个核心吸引子（Super Sinks），并输出其精确的物理坐标和权重。拒绝硬编码 (Hard-coding)。

---

## 1. 技术栈 (Tech Stack)
* **Language**: Python 3.9+
* **Core Libraries**:
    * `pandas`: 数据处理与聚合。
    * `geopandas`: 处理 Shapefile/GeoJSON 几何数据。
    * `scikit-learn`: 使用 DBSCAN 进行空间聚类。
    * `numpy`: 数值计算。
    * `matplotlib` / `contextily`: 结果可视化验证。

---

## 2. 项目文件结构 (Directory Structure)

请严格按照以下结构创建文件：

```text
geo-sfm-sim/
├── data/
│   ├── raw/
│   │   ├── mi_od_main_JT00_2021.csv.gz  # Census LODES OD 数据
│   │   └── tl_2021_26_tract.zip         # Census Tract Shapefile (密歇根州)
│   └── processed/
│       └── sinks_phase1.csv             # [输出] 最终计算出的 K 个核心汇
├── src/
│   ├── phase1/
│   │   ├── __init__.py
│   │   ├── config.py              # 配置参数 (阈值, 路径)
│   │   ├── data_loader.py         # 数据加载与预处理
│   │   ├── sink_identifier.py     # 核心算法 (Pareto + DBSCAN)
│   │   └── visualizer.py          # 绘图验证模块
├── main_phase1.py                 # Phase 1 执行入口
├── requirements.txt
└── DEVELOPMENT_PHASE1.md          # 本文档
````

-----

## 3\. 模块功能详述 (Module Specifications)

### A. `src/phase1/config.py`

**功能**：集中管理所有硬编码参数，方便调优。
**需包含的变量**：

  * `RAW_OD_PATH`:指向 `data/raw/` 下的 OD 文件路径。
  * `RAW_SHAPE_PATH`: 指向 `data/raw/` 下的 Shapefile 路径。
  * `PARETO_THRESHOLD`: **0.80** (累积流量截断阈值，即保留覆盖 80% 总流量的区域)。
  * `DBSCAN_EPS_KM`: **1.5** (聚类半径，单位公里。需转换为经纬度度数或投影坐标)。
  * `DBSCAN_MIN_SAMPLES`: **1** (只要有一个高流量点也可成为独立 Cluster)。
  * `OUTPUT_PATH`: `data/processed/sinks_phase1.csv`。

### B. `src/phase1/data_loader.py`

**功能**：负责读取原始数据并进行初步清洗。
**核心函数**：

1.  `load_od_data(filepath)`:
      * 读取 CSV。
      * 聚合：按 `w_tract` (Workplace Tract ID) `groupby`，求和 `S000` (Total Jobs)。
      * 返回：DataFrame `[tract_id, total_inflow]`。
2.  `load_geo_data(filepath)`:
      * 读取 Shapefile。
      * **关键**：计算每个 Tract 的几何中心 (Centroid)。
      * 返回：GeoDataFrame `[tract_id, geometry, centroid_lat, centroid_lon]`。

### C. `src/phase1/sink_identifier.py` (核心逻辑)

**功能**：执行“帕累托筛选”和“空间聚类”。
**核心函数**：

1.  `apply_pareto_filter(df_flow, threshold)`:
      * 输入：包含流量的 DataFrame。
      * 逻辑：按流量降序排列 -\> 计算 `cumsum` (累积和) -\> 计算 `cum_percentage`。
      * 操作：截取 `cum_percentage <= threshold` 的行。
      * 返回：筛选后的 High-Flow Tracts。
2.  `cluster_sinks(df_merged, eps_km, min_samples)`:
      * 输入：合并了经纬度的 High-Flow Tracts。
      * 逻辑：
          * 调用 `sklearn.cluster.DBSCAN`。
          * **注意**：如果是经纬度坐标，需使用 Haversine 距离度量，或者先投影到 UTM 坐标系再聚类。
      * **加权中心计算**：
          * 对于每个 Cluster，计算其包含的所有 Tracts 的**流量加权平均坐标**。
          * $X_{center} = \frac{\sum (x_i \cdot flow_i)}{\sum flow_i}$
      * 返回：包含 $K$ 个 Super Sinks 的 DataFrame `[cluster_id, lat, lon, total_weight]`。

### D. `src/phase1/visualizer.py`

**功能**：生成可视化图片，用于人工验证算法是否合理。
**核心函数**：

1.  `plot_sinks_on_map(gdf_tracts, df_sinks, output_img_path)`:
      * 底图：绘制所有 Census Tracts 的轮廓。
      * 热点：用散点图绘制识别出的 $K$ 个 Sinks。
      * 样式：点的大小 (Size) 应正比于流量权重。
      * 输出：保存为 PNG 图片。

### E. `main_phase1.py`

**功能**：串联整个流程。
**流程逻辑**：

1.  加载 Config。
2.  `data_loader.load_od_data` & `load_geo_data`。
3.  Merge 两个数据表（通过 GeoID/TractID）。
4.  `sink_identifier.apply_pareto_filter` 筛选头部区域。
5.  `sink_identifier.cluster_sinks` 执行空间聚类，得到核心汇。
6.  保存结果到 `data/processed/`。
7.  调用 `visualizer` 生成验证图。
8.  打印日志：显示识别出了多少个 Cluster，覆盖了多少流量。

-----

## 4\. 开发步骤 (Step-by-Step Instructions)

请 Codex 按照以下顺序编写代码：

1.  **环境配置**：创建 `config.py`，确保路径正确。
2.  **数据层实现**：编写 `data_loader.py`，确保能正确读取压缩的 Census 数据并提取几何中心。*提示：注意 Census ID 的字符串补零问题。*
3.  **核心算法实现**：编写 `sink_identifier.py`。先实现 Pareto 过滤，打印过滤后的数量；再实现 DBSCAN，打印聚类后的数量。
4.  **可视化实现**：编写 `visualizer.py`。
5.  **集成测试**：编写 `main_phase1.py`，运行全流程，并检查输出的 CSV 文件格式是否符合：`cluster_id, lat, lon, flow_weight`。

-----

## 5\. 预期输出 (Deliverables)

运行结束后，应在 `data/processed/` 目录下看到：

1.  **`sinks_phase1.csv`**:
    ```csv
    cluster_id,lat,lon,total_flow
    0,42.3314,-83.0458,54000  <-- (示例：底特律 CBD)
    1,42.2808,-83.7430,32000  <-- (示例：安娜堡)
    ...
    ```
2.  **`sinks_visualization.png`**: 一张地图，显示了密歇根州的轮廓以及高亮的 $K$ 个核心就业中心。

-----

```
```