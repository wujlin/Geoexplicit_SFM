"""
数据加载与预处理：
- 读取 OD CSV 并聚合 workplace tract 的总流量
- 读取 Tract Shapefile 并计算质心
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import pandas as pd

TRACT_ID_CANDIDATES = ["w_tract", "w_geocode", "work", "GEOID", "GEOID20", "geoid"]


def _choose_column(columns, candidates) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"找不到 tract/geoid 列，候选 {candidates}，实际列：{list(columns)}")


def _normalize_geoid(series: pd.Series) -> pd.Series:
    """确保 Tract ID 是 11 位字符串。"""
    return series.astype(str).str.zfill(11)


def load_od_data(filepath: Path) -> pd.DataFrame:
    """
    读取 OD CSV，按 workplace tract 聚合总流量。
    预期包含列: w_tract / w_geocode 以及 S000（Total Jobs）。
    返回列: tract_id, total_inflow
    """
    df = pd.read_csv(filepath)
    tract_col = _choose_column(df.columns, ["w_tract", "w_geocode", "work"])
    flow_col = _choose_column(df.columns, ["S000", "s000", "TOTAL_JOBS"])  # 兼容大小写

    df["tract_id"] = _normalize_geoid(df[tract_col])
    agg = df.groupby("tract_id", as_index=False)[flow_col].sum()
    agg = agg.rename(columns={flow_col: "total_inflow"})
    return agg[["tract_id", "total_inflow"]]


def load_geo_data(
    filepath: Path,
    target_crs: str = "EPSG:4326",
    projected_crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """
    读取 Tract Shapefile/zip，计算质心，输出列: tract_id, geometry, centroid_lat, centroid_lon
    """
    gdf = gpd.read_file(filepath)
    tract_col = _choose_column(gdf.columns, ["GEOID", "GEOID20", "geoid"])
    gdf = gdf.to_crs(target_crs)
    gdf["tract_id"] = _normalize_geoid(gdf[tract_col])

    # 使用投影坐标计算质心，避免地理 CRS 警告
    gdf_proj = gdf.to_crs(projected_crs)
    centroids_proj = gdf_proj.geometry.centroid
    centroids = gpd.GeoSeries(centroids_proj, crs=projected_crs).to_crs(target_crs)
    gdf["centroid"] = centroids
    gdf["centroid_lon"] = centroids.x
    gdf["centroid_lat"] = centroids.y
    return gdf[["tract_id", "geometry", "centroid_lat", "centroid_lon"]]


def merge_flow_with_geo(df_flow: pd.DataFrame, gdf_geo: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """合并流量与地理信息。"""
    merged = gdf_geo.merge(df_flow, on="tract_id", how="inner")
    return merged


def filter_by_counties(df: pd.DataFrame, county_fips: list[str], tract_col: str = "tract_id") -> pd.DataFrame:
    """按 county 前 5 位 FIPS 过滤 tract。"""
    if not county_fips:
        return df
    return df[df[tract_col].str[:5].isin(county_fips)].copy()


__all__: Tuple[str, ...] = ("load_od_data", "load_geo_data", "merge_flow_with_geo", "filter_by_counties")
