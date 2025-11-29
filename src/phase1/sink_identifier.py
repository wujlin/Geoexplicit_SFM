"""
帕累托筛选 + DBSCAN 聚类识别核心汇。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0088


def apply_pareto_filter(df_flow: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    按流量降序，截取覆盖 threshold (e.g., 0.8) 的高流量区域。
    """
    df_sorted = df_flow.sort_values("total_inflow", ascending=False).reset_index(drop=True)
    total = df_sorted["total_inflow"].sum()
    df_sorted["cum_ratio"] = df_sorted["total_inflow"].cumsum() / (total + 1e-9)
    filtered = df_sorted[df_sorted["cum_ratio"] <= threshold].copy()
    return filtered.drop(columns=["cum_ratio"])


def _haversine_dbscan(coords_latlon_deg: np.ndarray, eps_km: float, min_samples: int) -> np.ndarray:
    """
    使用 Haversine 距离的 DBSCAN；输入为 (n,2) 的经纬度（度）。
    """
    coords_rad = np.radians(coords_latlon_deg)
    eps_rad = eps_km / EARTH_RADIUS_KM
    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(coords_rad)
    return labels


def cluster_sinks(df_highflow: pd.DataFrame, eps_km: float, min_samples: int) -> pd.DataFrame:
    """
    对帕累托筛选后的点进行空间聚类，输出每个 cluster 的流量加权中心。
    需要列: centroid_lat, centroid_lon, total_inflow
    返回列: cluster_id, lat, lon, total_flow
    """
    coords = df_highflow[["centroid_lat", "centroid_lon"]].to_numpy()
    labels = _haversine_dbscan(coords, eps_km=eps_km, min_samples=min_samples)
    df_highflow = df_highflow.copy()
    df_highflow["cluster"] = labels

    clusters = []
    cluster_ids = sorted([c for c in np.unique(labels) if c != -1])  # DBSCAN 噪声为 -1
    for new_id, cluster_label in enumerate(cluster_ids):
        df_c = df_highflow[df_highflow["cluster"] == cluster_label]
        weight = df_c["total_inflow"].to_numpy()
        lats = df_c["centroid_lat"].to_numpy()
        lons = df_c["centroid_lon"].to_numpy()
        wsum = weight.sum()
        lat_center = (lats * weight).sum() / wsum
        lon_center = (lons * weight).sum() / wsum
        clusters.append(
            {
                "cluster_id": new_id,
                "lat": lat_center,
                "lon": lon_center,
                "total_flow": wsum,
            }
        )

    return pd.DataFrame(clusters)


__all__ = ["apply_pareto_filter", "cluster_sinks"]
