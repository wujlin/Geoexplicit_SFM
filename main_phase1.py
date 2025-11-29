from pathlib import Path

import pandas as pd

from src.phase1 import config
from src.phase1.data_loader import load_geo_data, load_od_data, merge_flow_with_geo
from src.phase1.sink_identifier import apply_pareto_filter, cluster_sinks
from src.phase1.visualizer import plot_sinks_on_map


def ensure_output_dir():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_output_dir()
    print(f"读取 OD 数据：{config.RAW_OD_PATH}")
    df_flow = load_od_data(config.RAW_OD_PATH)
    print(f"OD 记录数：{len(df_flow)}")

    print(f"读取地理数据：{config.RAW_SHAPE_PATH}")
    gdf_geo = load_geo_data(config.RAW_SHAPE_PATH, target_crs=config.TARGET_CRS)
    print(f"Tract 数量：{len(gdf_geo)}")

    merged = merge_flow_with_geo(df_flow, gdf_geo)
    print(f"合并后记录：{len(merged)}")

    high = apply_pareto_filter(merged[["tract_id", "total_inflow", "centroid_lat", "centroid_lon"]], config.PARETO_THRESHOLD)
    print(f"帕累托筛选后：{len(high)} 条（覆盖 {config.PARETO_THRESHOLD*100:.0f}% 总流量）")

    sinks = cluster_sinks(high, eps_km=config.DBSCAN_EPS_KM, min_samples=config.DBSCAN_MIN_SAMPLES)
    print(f"聚类得到 {len(sinks)} 个核心汇")

    sinks.to_csv(config.OUTPUT_SINKS_CSV, index=False)
    print(f"已保存：{config.OUTPUT_SINKS_CSV}")

    plot_sinks_on_map(gdf_geo, sinks, config.OUTPUT_VIZ, figsize=config.FIG_SIZE)
    print(f"已保存可视化：{config.OUTPUT_VIZ}")


if __name__ == "__main__":
    main()
