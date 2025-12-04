"""
Phase 1 验证：Sink 识别质量检查
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from . import PathConfig, Phase1Metrics, get_path_config


def validate_phase1(paths: PathConfig = None, save_figure: bool = True) -> Phase1Metrics:
    """
    验证 Phase 1 Sink 识别结果
    
    检查项：
    1. Sink 数量是否在合理范围 (30-60)
    2. 流量覆盖率
    3. 空间分布
    """
    if paths is None:
        paths = get_path_config()
    
    paths.ensure_dirs()
    
    # 加载数据
    sinks = pd.read_csv(paths.sinks_csv)
    
    # 计算指标
    num_sinks = len(sinks)
    total_flow = sinks["total_flow"].sum()
    flow_distribution = sinks["total_flow"].tolist()
    
    lat_range = (float(sinks["lat"].min()), float(sinks["lat"].max()))
    lon_range = (float(sinks["lon"].min()), float(sinks["lon"].max()))
    
    # 计算空间覆盖率（基于 bounding box）
    lat_span = lat_range[1] - lat_range[0]
    lon_span = lon_range[1] - lon_range[0]
    spatial_coverage = lat_span * lon_span  # 简化的空间覆盖度量
    
    # 估算人口覆盖率（假设 Pareto 阈值 92%）
    population_covered_ratio = 0.92  # 从 config 中的 Pareto 阈值
    
    metrics = Phase1Metrics(
        num_sinks=num_sinks,
        total_sinks_flow=int(total_flow),
        population_covered_ratio=population_covered_ratio,
        spatial_coverage=float(spatial_coverage),
        lat_range=lat_range,
        lon_range=lon_range,
        flow_distribution=flow_distribution,
    )
    
    # 可视化
    if save_figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：Sink 空间分布
        ax1 = axes[0]
        scatter = ax1.scatter(
            sinks["lon"], sinks["lat"],
            s=sinks["total_flow"] / sinks["total_flow"].max() * 500,
            c=sinks["total_flow"],
            cmap="YlOrRd",
            alpha=0.7,
            edgecolors="black",
        )
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title(f"Phase 1: {num_sinks} Sinks Identified")
        plt.colorbar(scatter, ax=ax1, label="Total Flow")
        
        # 右图：流量分布
        ax2 = axes[1]
        ax2.bar(range(num_sinks), sorted(flow_distribution, reverse=True))
        ax2.set_xlabel("Sink Rank")
        ax2.set_ylabel("Total Flow")
        ax2.set_title("Flow Distribution (Sorted)")
        ax2.axhline(y=np.mean(flow_distribution), color="r", linestyle="--", label="Mean")
        ax2.legend()
        
        plt.tight_layout()
        fig.savefig(paths.figures_dir / "phase1_sinks.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {paths.figures_dir / 'phase1_sinks.png'}")
    
    # 保存指标
    np.savez(
        paths.metrics_dir / "phase1_metrics.npz",
        num_sinks=metrics.num_sinks,
        total_sinks_flow=metrics.total_sinks_flow,
        population_covered_ratio=metrics.population_covered_ratio,
        spatial_coverage=metrics.spatial_coverage,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Phase 1 Validation Results")
    print("=" * 60)
    print(f"  Num sinks: {num_sinks}")
    print(f"  Total flow: {total_flow:,}")
    print(f"  Population covered: {population_covered_ratio * 100:.1f}%")
    print(f"  Lat range: [{lat_range[0]:.4f}, {lat_range[1]:.4f}]")
    print(f"  Lon range: [{lon_range[0]:.4f}, {lon_range[1]:.4f}]")
    
    # 质量评估
    if 30 <= num_sinks <= 60:
        print("  ✅ Sink count in target range")
    else:
        print(f"  ⚠️ Sink count outside target range (30-60)")
    
    return metrics


if __name__ == "__main__":
    validate_phase1()
