"""
可视化：绘制 Tract 边界与识别出的核心汇。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from shapely.geometry import Point

try:
    import contextily as cx
except ImportError:  # pragma: no cover - optional runtime check
    cx = None


def plot_sinks_on_map(gdf_tracts, df_sinks, output_img_path, figsize=(10, 10)):
    """
    绘制底图 + sinks 散点，点大小按 total_flow 映射。
    """
    fig, ax = plt.subplots(figsize=figsize)
    gdf_web = gdf_tracts.to_crs(epsg=3857)
    gdf_web.plot(ax=ax, color="none", edgecolor="#98a2b3", linewidth=0.4, alpha=0.5)

    sinks_geo = df_sinks.copy()
    sinks_geo_gdf = gdf_web.iloc[:0].copy()  # empty GeoDataFrame with CRS
    sinks_geo_gdf["geometry"] = [Point(lon, lat) for lat, lon in zip(sinks_geo["lat"], sinks_geo["lon"])]
    sinks_geo_gdf["total_flow"] = sinks_geo["total_flow"].to_numpy()
    sinks_geo_gdf = sinks_geo_gdf.set_crs(epsg=4326).to_crs(epsg=3857)

    max_flow = max(float(sinks_geo_gdf["total_flow"].max()), 1.0)
    sizes = (sinks_geo_gdf["total_flow"] / max_flow) * 400 + 50
    sinks_geo_gdf.plot(
        ax=ax,
        markersize=sizes,
        color="#ef4444",
        alpha=0.7,
        edgecolor="#1f2937",
        linewidth=0.8,
    )

    if cx:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=9, alpha=0.9)
        except Exception:
            pass

    ax.set_axis_off()
    ax.set_title("Identified Super Sinks (Phase 1)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_img_path, dpi=200)
    plt.close(fig)


__all__ = ["plot_sinks_on_map"]
