from pathlib import Path

import pandas as pd

from src.phase1 import config
from src.phase1.data_loader import filter_by_counties, load_geo_data, load_od_data, merge_flow_with_geo
from src.phase1.sink_identifier import apply_pareto_filter, cluster_sinks, filter_by_flow_threshold
from src.phase1.visualizer import plot_sinks_on_map


def ensure_output_dir():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_output_dir()
    print(f"读取 OD 数据：{config.RAW_OD_PATH}")
    df_flow = load_od_data(config.RAW_OD_PATH)
    print(f"OD 记录数（聚合前行数在原始 CSV）：{len(pd.read_csv(config.RAW_OD_PATH))}")
    print(f"工作地唯一 tract 数：{df_flow['tract_id'].nunique()}")

    df_flow = filter_by_counties(df_flow, config.TARGET_COUNTIES, tract_col="tract_id")
    print(f"按目标县过滤后工作地 tract 数：{df_flow['tract_id'].nunique()}")
    print(f"过滤后总流量：{df_flow['total_inflow'].sum():.0f}")

    print(f"读取地理数据：{config.RAW_SHAPE_PATH}")
    gdf_geo = load_geo_data(
        config.RAW_SHAPE_PATH,
        target_crs=config.TARGET_CRS,
        projected_crs=config.PROJECTED_CRS,
    )
    print(f"Tract 数量：{len(gdf_geo)}")

    merged = merge_flow_with_geo(df_flow, gdf_geo)
    print(f"合并后记录：{len(merged)}")

    high = apply_pareto_filter(merged[["tract_id", "total_inflow", "centroid_lat", "centroid_lon"]], config.PARETO_THRESHOLD)
    print(f"帕累托筛选后：{len(high)} 条（覆盖 {config.PARETO_THRESHOLD*100:.0f}% 总流量）")

    high, flow_thr = filter_by_flow_threshold(high, quantile=config.FLOW_MIN_QUANTILE, min_value=config.FLOW_MIN_VALUE)
    if flow_thr is not None:
        print(f"额外流量阈值过滤：>= {flow_thr:.0f}，剩余 {len(high)} 条")

    sinks = cluster_sinks(high, eps_km=config.DBSCAN_EPS_KM, min_samples=config.DBSCAN_MIN_SAMPLES)
    print(f"聚类得到 {len(sinks)} 个核心汇，平均每簇 {sinks['n_points'].mean():.2f} 个点，最大簇 {sinks['n_points'].max()} 个点")

    # 诊断输出：前 K 簇覆盖率
    total_flow_all = sinks["total_flow"].sum()
    topk = sinks.sort_values("total_flow", ascending=False).head(config.LOG_TOP_K)
    topk_sum = topk["total_flow"].sum()
    print(f"前 {config.LOG_TOP_K} 个簇覆盖 {topk_sum/total_flow_all*100:.1f}% 聚类总流量")
    print(topk[["cluster_id", "total_flow", "n_points"]])

    sinks.to_csv(config.OUTPUT_SINKS_CSV, index=False)
    print(f"已保存：{config.OUTPUT_SINKS_CSV}")

    plot_sinks_on_map(gdf_geo, sinks, config.OUTPUT_VIZ, figsize=config.FIG_SIZE, use_basemap=config.USE_BASEMAP)
    print(f"已保存可视化：{config.OUTPUT_VIZ}")


if __name__ == "__main__":
    main()
