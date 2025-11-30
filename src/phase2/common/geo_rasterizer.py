"""
路网栅格化与目标密度生成：
- 从 Phase1 的 sinks 计算 bbox
- 下载/加载 OSM 路网，栅格化为 walkable mask
- 将 sinks 高斯核叠加到栅格生成 target density（可选融合 worldpop）
"""

from __future__ import annotations

import math
import inspect
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from rasterio import features
from rasterio import warp
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from src.phase2 import config


def _load_sinks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("sinks CSV 需要包含 lat, lon 列")
    if config.SINK_WEIGHT_KEY not in df.columns:
        raise ValueError(f"sinks CSV 缺少权重列 {config.SINK_WEIGHT_KEY}")
    return df


def _deg_per_km(lat: float) -> Tuple[float, float]:
    # 粗略换算，足够用于 bbox padding
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.320 * math.cos(math.radians(lat)) + 1e-9)
    return lat_per_km, lon_per_km


def _geoms_to_mask(geoms, bounds_proj, resolution_m: float) -> Tuple[np.ndarray, rasterio.Affine]:
    """通用几何栅格化，输入为投影坐标下的 Line/Pseudo geometry。"""
    minx, miny, maxx, maxy = bounds_proj
    width = int(math.ceil((maxx - minx) / resolution_m))
    height = int(math.ceil((maxy - miny) / resolution_m))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    shapes = ((geom, 1) for geom in geoms if geom is not None)
    mask = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return mask, transform


def compute_bbox(sinks: pd.DataFrame, padding_km: float) -> Tuple[float, float, float, float]:
    lat0 = sinks["lat"].mean()
    lat_per_km, lon_per_km = _deg_per_km(lat0)
    pad_lat = padding_km * lat_per_km
    pad_lon = padding_km * lon_per_km
    north = sinks["lat"].max() + pad_lat
    south = sinks["lat"].min() - pad_lat
    east = sinks["lon"].max() + pad_lon
    west = sinks["lon"].min() - pad_lon
    return north, south, east, west


def load_local_edges(bbox: Tuple[float, float, float, float], shp_path: Optional[Path]):
    """
    从本地路网 Shapefile 读取 bbox 范围内的线数据。
    """
    if shp_path is None:
        return None
    shp_path = Path(shp_path)
    if not shp_path.exists():
        if config.VERBOSE:
            print(f"local road shapefile not found: {shp_path}")
        return None
    north, south, east, west = bbox
    try:
        gdf = gpd.read_file(shp_path, bbox=(west, south, east, north))
    except Exception as exc:  # pragma: no cover - IO error path
        if config.VERBOSE:
            print(f"failed to read local road shapefile: {exc}")
        return None
    if gdf.empty:
        if config.VERBOSE:
            print("local road shapefile returned empty for bbox")
        return None
    return gdf


def load_osm_graph(bbox: Tuple[float, float, float, float], cache_path: Path) -> ox.graph.Graph:
    """
    加载/下载 OSM 图；优先使用 cache。
    """
    if cache_path.exists():
        return ox.load_graphml(cache_path)
    north, south, east, west = bbox
    sig = inspect.signature(ox.graph_from_bbox)
    params = list(sig.parameters)
    # 兼容不同版本 osmnx：有的要求 bbox 作为单参，有的支持 north/south/east/west
    if len(params) == 1 or "bbox" in params:
        G = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type=config.NETWORK_TYPE,
            simplify=True,
        )
    else:
        G = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type=config.NETWORK_TYPE,
            simplify=True,
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, cache_path)
    return G


def _graph_to_mask(G, bounds_proj, resolution_m: float) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    将投影坐标系下的路网转换为二值掩膜。
    """
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    mask, transform = _geoms_to_mask(edges.geometry, bounds_proj, resolution_m)
    return mask, transform


def _sinks_to_density(sinks: pd.DataFrame, transform: rasterio.Affine, shape: Tuple[int, int], sigma_m: float) -> np.ndarray:
    """
    将 sinks 投影到栅格并做高斯模糊。
    """
    height, width = shape
    density = np.zeros((height, width), dtype=float)
    sigma_pix = sigma_m / config.GRID_RES_M
    for _, row in sinks.iterrows():
        # 转换地理坐标到像素索引
        col, rowpix = ~transform * (row["lon_proj"], row["lat_proj"])
        r = int(round(rowpix))
        c = int(round(col))
        if 0 <= r < height and 0 <= c < width:
            density[r, c] += float(row[config.SINK_WEIGHT_KEY])
    density = gaussian_filter(density, sigma=sigma_pix)
    return density


def _maybe_merge_worldpop(
    density: np.ndarray,
    transform: rasterio.Affine,
    worldpop_path: Optional[Path],
    dst_crs=None,
) -> np.ndarray:
    if worldpop_path is None:
        return density
    try:
        with rasterio.open(worldpop_path) as src:
            wp = src.read(1, masked=True)
            wp_resampled = warp.reproject(
                source=wp,
                destination=np.empty_like(density, dtype=float),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs or src.crs,
                resampling=warp.Resampling.bilinear,
            )[0]
            wp_resampled = np.nan_to_num(wp_resampled, nan=0.0)
            # 简单归一化后叠加
            wp_resampled /= (wp_resampled.max() + 1e-9)
            density = density + wp_resampled * density.max()
    except Exception:
        # 若失败，不阻塞流程
        pass
    return density


def build_walkable_and_density(
    sinks_path: Path = config.SINKS_PATH,
    osm_cache: Path = config.OSM_CACHE,
    worldpop_path: Optional[Path] = config.WORLDPOP_RASTER,
) -> Tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    """
    主流程：读取 sinks -> bbox -> OSM -> 掩膜 -> 密度
    返回：mask, density, transform
    """
    sinks = _load_sinks(sinks_path)
    bbox = compute_bbox(sinks, config.BBOX_PADDING_KM)
    edges_local = load_local_edges(bbox, config.LOCAL_ROAD_PATH)

    if edges_local is not None:
        # 使用本地路网，避免联网
        edges_proj = edges_local.to_crs(config.PROJECTED_CRS)
        bounds_proj = edges_proj.total_bounds
        mask, transform = _geoms_to_mask(edges_proj.geometry, bounds_proj, config.GRID_RES_M)
        target_crs = edges_proj.crs
        if config.VERBOSE:
            print(f"using local road shapefile: {config.LOCAL_ROAD_PATH}")
    else:
        # 回退到 OSM 拉取
        G = load_osm_graph(bbox, osm_cache)
        G_proj = ox.project_graph(G)
        bounds_proj = ox.utils_geo.graph_bounds(G_proj)
        mask, transform = _graph_to_mask(G_proj, bounds_proj, config.GRID_RES_M)
        nodes, _ = ox.graph_to_gdfs(G_proj, nodes=True, edges=False)
        target_crs = nodes.crs

    # 路网覆盖检查
    mask_ratio = mask.mean()
    if config.VERBOSE:
        print(f"walkable mask ratio: {mask_ratio:.4f}")
    if mask_ratio < config.MIN_MASK_FRACTION:
        print("warning: walkable mask coverage is very low; check bbox or network_type")

    # sinks 投影到图的 CRS
    sinks_gdf = gpd.GeoDataFrame(
        sinks,
        geometry=gpd.points_from_xy(sinks["lon"], sinks["lat"]),
        crs=config.TARGET_CRS,
    ).to_crs(target_crs)
    sinks["lon_proj"] = sinks_gdf.geometry.x
    sinks["lat_proj"] = sinks_gdf.geometry.y

    density = _sinks_to_density(sinks, transform, mask.shape, config.SINK_GAUSS_SIGMA_M)
    density = _maybe_merge_worldpop(density, transform, worldpop_path, dst_crs=target_crs)

    # 归一化
    density = density / (density.max() + 1e-9)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.WALKABLE_MASK_PATH, mask)
    np.save(config.TARGET_DENSITY_PATH, density)
    if config.VERBOSE:
        print(f"saved mask to {config.WALKABLE_MASK_PATH}")
        print(f"saved density to {config.TARGET_DENSITY_PATH}")
    return mask, density, transform


if __name__ == "__main__":
    build_walkable_and_density()
