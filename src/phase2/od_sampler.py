"""
OD 采样模块：
1. 构建 tract→pixel 映射
2. 构建 tract→sink 映射（基于空间最近邻）
3. 聚合 OD 流到 sink 级别，生成采样表

核心输出：
- tract_pixel_mapping.csv: tract GEOID -> (px, py) 像素坐标
- tract_sink_mapping.csv: tract GEOID -> sink_id
- sink_od_matrix.csv: (origin_sink_id, dest_sink_id) -> flow
- sink_od_prob.csv: origin_pixel (px, py) -> dest_sink_id 的采样概率
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

from src.phase2 import config


def load_raster_meta():
    """加载栅格元数据"""
    meta_path = config.OUTPUT_DIR / "raster_meta.json"
    with open(meta_path) as f:
        return json.load(f)


def build_tract_pixel_mapping(
    tract_path: Path = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    构建 tract GEOID → 像素坐标 (px, py) 映射
    
    Args:
        tract_path: tract shapefile 路径
        output_path: 输出 CSV 路径
        
    Returns:
        DataFrame with columns: GEOID, cx, cy, px, py (只包含栅格范围内的 tract)
    """
    if tract_path is None:
        tract_path = Path("dataset/geo/mi-tracts-demo-work")
    if output_path is None:
        output_path = config.OUTPUT_DIR / "tract_pixel_mapping.csv"
    
    # 加载元数据
    meta = load_raster_meta()
    minx, miny, maxx, maxy = meta["bounds_proj"]
    res = meta["grid_res_m"]
    H, W = meta["shape"]
    
    print(f"[tract_pixel_mapping] Grid: {H}x{W}, res={res}m")
    
    # 读取 tract 数据
    tracts = gpd.read_file(tract_path)
    print(f"  Total tracts: {len(tracts)}")
    
    # 投影到栅格 CRS (EPSG:3857)
    tracts_proj = tracts.to_crs(meta["crs"])
    
    # 计算质心
    tracts_proj["centroid"] = tracts_proj.geometry.centroid
    tracts_proj["cx"] = tracts_proj["centroid"].x
    tracts_proj["cy"] = tracts_proj["centroid"].y
    
    # 转换为像素坐标
    # 注意：栅格原点在左上角，y 轴向下
    tracts_proj["px"] = (tracts_proj["cx"] - minx) / res
    tracts_proj["py"] = (maxy - tracts_proj["cy"]) / res
    
    # 筛选在栅格范围内的 tract
    in_bounds = (
        (tracts_proj["px"] >= 0) & (tracts_proj["px"] < W) &
        (tracts_proj["py"] >= 0) & (tracts_proj["py"] < H)
    )
    
    mapping = tracts_proj[["GEOID", "cx", "cy", "px", "py"]][in_bounds].copy()
    mapping["px"] = mapping["px"].astype(float)
    mapping["py"] = mapping["py"].astype(float)
    
    print(f"  Tracts in grid bounds: {len(mapping)}")
    
    # 保存
    mapping.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    
    return mapping


def build_tract_sink_mapping(
    tract_mapping: pd.DataFrame = None,
    sinks_path: Path = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    构建 tract GEOID → sink_id 映射（基于空间最近邻）
    
    Args:
        tract_mapping: tract→pixel 映射 DataFrame
        sinks_path: sinks CSV 路径
        output_path: 输出 CSV 路径
        
    Returns:
        DataFrame with columns: GEOID, sink_id, dist_to_sink
    """
    if tract_mapping is None:
        tract_mapping = pd.read_csv(config.OUTPUT_DIR / "tract_pixel_mapping.csv")
    if sinks_path is None:
        sinks_path = config.SINKS_PATH
    if output_path is None:
        output_path = config.OUTPUT_DIR / "tract_sink_mapping.csv"
    
    # 加载元数据
    meta = load_raster_meta()
    minx, miny, maxx, maxy = meta["bounds_proj"]
    res = meta["grid_res_m"]
    
    # 读取 sinks
    sinks = pd.read_csv(sinks_path)
    print(f"[tract_sink_mapping] Sinks: {len(sinks)}")
    
    # 将 sinks 投影到 EPSG:3857
    sinks_gdf = gpd.GeoDataFrame(
        sinks,
        geometry=gpd.points_from_xy(sinks["lon"], sinks["lat"]),
        crs="EPSG:4326",
    ).to_crs(meta["crs"])
    
    sinks["sx"] = sinks_gdf.geometry.x
    sinks["sy"] = sinks_gdf.geometry.y
    
    # 计算 sinks 像素坐标
    sinks["spx"] = (sinks["sx"] - minx) / res
    sinks["spy"] = (maxy - sinks["sy"]) / res
    
    print(f"  Sinks px range: [{sinks['spx'].min():.1f}, {sinks['spx'].max():.1f}]")
    print(f"  Sinks py range: [{sinks['spy'].min():.1f}, {sinks['spy'].max():.1f}]")
    
    # 使用 KD-Tree 找最近邻 sink
    sink_coords = sinks[["spx", "spy"]].values
    tract_coords = tract_mapping[["px", "py"]].values
    
    tree = cKDTree(sink_coords)
    distances, indices = tree.query(tract_coords, k=1)
    
    # 构建映射
    result = tract_mapping[["GEOID"]].copy()
    result["sink_id"] = sinks.iloc[indices]["cluster_id"].values
    result["dist_to_sink"] = distances * res  # 转换为米
    
    print(f"  Mean dist to sink: {result['dist_to_sink'].mean():.0f}m")
    print(f"  Max dist to sink: {result['dist_to_sink'].max():.0f}m")
    
    # 保存
    result.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    
    return result


def aggregate_od_to_sink_level(
    od_path: Path = None,
    tract_sink_mapping: pd.DataFrame = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    将 tract-level OD 流聚合到 sink 级别
    
    Args:
        od_path: OD 流 CSV 路径
        tract_sink_mapping: tract→sink 映射
        output_path: 输出 CSV 路径
        
    Returns:
        DataFrame with columns: origin_sink, dest_sink, flow
    """
    if od_path is None:
        od_path = Path("dataset/od_flow/mi-tract-od-2020.csv")
    if tract_sink_mapping is None:
        tract_sink_mapping = pd.read_csv(config.OUTPUT_DIR / "tract_sink_mapping.csv")
    if output_path is None:
        output_path = config.OUTPUT_DIR / "sink_od_matrix.csv"
    
    # 读取 OD 流数据
    od = pd.read_csv(od_path)
    print(f"[aggregate_od] OD flows: {len(od)}")
    
    # 确保 GEOID 格式一致（字符串）
    tract_sink_mapping["GEOID"] = tract_sink_mapping["GEOID"].astype(str)
    od["home"] = od["home"].astype(str)
    od["work"] = od["work"].astype(str)
    
    # 构建 GEOID → sink_id 字典
    geoid_to_sink = dict(zip(tract_sink_mapping["GEOID"], tract_sink_mapping["sink_id"]))
    
    # 映射 origin (home) 和 destination (work) 到 sink_id
    od["origin_sink"] = od["home"].map(geoid_to_sink)
    od["dest_sink"] = od["work"].map(geoid_to_sink)
    
    # 过滤掉无法映射的记录
    valid = od["origin_sink"].notna() & od["dest_sink"].notna()
    od_valid = od[valid].copy()
    print(f"  OD flows with valid mapping: {len(od_valid)} ({len(od_valid)/len(od)*100:.1f}%)")
    
    # 聚合到 sink 级别
    sink_od = od_valid.groupby(["origin_sink", "dest_sink"])["S000"].sum().reset_index()
    sink_od.columns = ["origin_sink", "dest_sink", "flow"]
    sink_od["origin_sink"] = sink_od["origin_sink"].astype(int)
    sink_od["dest_sink"] = sink_od["dest_sink"].astype(int)
    
    print(f"  Sink-level OD pairs: {len(sink_od)}")
    print(f"  Total flow: {sink_od['flow'].sum():,.0f}")
    
    # 保存
    sink_od.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    
    return sink_od


def build_origin_dest_prob_table(
    sink_od: pd.DataFrame = None,
    tract_sink_mapping: pd.DataFrame = None,
    tract_pixel_mapping: pd.DataFrame = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    为每个 origin pixel 构建目的地 sink 的采样概率表
    
    逻辑：
    1. 每个 tract 属于一个 origin_sink
    2. 该 origin_sink 出发的 OD 流构成目的地 sink 的概率分布
    3. 同一 origin_sink 内所有 tract 共享相同的目的地概率分布
    
    Args:
        sink_od: sink-level OD 矩阵
        tract_sink_mapping: tract→sink 映射
        tract_pixel_mapping: tract→pixel 映射
        output_path: 输出路径
        
    Returns:
        DataFrame with columns: origin_sink, dest_sink, prob
    """
    if sink_od is None:
        sink_od = pd.read_csv(config.OUTPUT_DIR / "sink_od_matrix.csv")
    if tract_sink_mapping is None:
        tract_sink_mapping = pd.read_csv(config.OUTPUT_DIR / "tract_sink_mapping.csv")
    if tract_pixel_mapping is None:
        tract_pixel_mapping = pd.read_csv(config.OUTPUT_DIR / "tract_pixel_mapping.csv")
    if output_path is None:
        output_path = config.OUTPUT_DIR / "sink_od_prob.csv"
    
    print(f"[build_origin_dest_prob_table]")
    
    # 计算每个 origin_sink 出发的目的地概率分布
    origin_total = sink_od.groupby("origin_sink")["flow"].sum()
    sink_od["prob"] = sink_od.apply(
        lambda row: row["flow"] / origin_total[row["origin_sink"]] 
        if row["origin_sink"] in origin_total.index else 0,
        axis=1
    )
    
    # 提取概率表
    prob_table = sink_od[["origin_sink", "dest_sink", "prob"]].copy()
    
    # 验证概率和
    prob_sum = prob_table.groupby("origin_sink")["prob"].sum()
    print(f"  Origin sinks: {len(prob_sum)}")
    print(f"  Prob sum check: min={prob_sum.min():.6f}, max={prob_sum.max():.6f}")
    
    # 保存
    prob_table.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    
    return prob_table


def build_all_mappings():
    """运行完整的映射构建流程"""
    print("=" * 60)
    print("Building OD sampling tables")
    print("=" * 60)
    
    # Step 1: tract → pixel
    tract_pixel = build_tract_pixel_mapping()
    
    # Step 2: tract → sink
    tract_sink = build_tract_sink_mapping(tract_pixel)
    
    # Step 3: aggregate OD to sink level
    sink_od = aggregate_od_to_sink_level(tract_sink_mapping=tract_sink)
    
    # Step 4: build probability table
    prob_table = build_origin_dest_prob_table(
        sink_od=sink_od,
        tract_sink_mapping=tract_sink,
        tract_pixel_mapping=tract_pixel,
    )
    
    print("=" * 60)
    print("Done!")
    
    return {
        "tract_pixel": tract_pixel,
        "tract_sink": tract_sink,
        "sink_od": sink_od,
        "prob_table": prob_table,
    }


if __name__ == "__main__":
    build_all_mappings()
