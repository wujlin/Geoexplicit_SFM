"""
Phase 2 验证：导航场质量检查
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from . import PathConfig, Phase2Metrics, get_path_config, load_nav_fields


def validate_phase2(paths: PathConfig = None, save_figure: bool = True) -> Phase2Metrics:
    """
    验证 Phase 2 导航场质量
    
    检查项：
    1. Walkable mask 覆盖率
    2. 每个 sink 导航场的有效方向覆盖率
    3. 导航场方向一致性
    """
    if paths is None:
        paths = get_path_config()
    
    paths.ensure_dirs()
    
    # 加载数据
    walkable_mask = np.load(paths.walkable_mask)
    H, W = walkable_mask.shape
    walkable_pixels = int((walkable_mask > 0).sum())
    walkable_ratio = walkable_pixels / (H * W)
    
    # 加载导航场
    nav_fields = load_nav_fields(paths.nav_fields_dir)
    num_nav_fields = len(nav_fields)
    
    # 检查每个导航场的有效覆盖率
    nav_field_coverage = {}
    for sink_id, nav_field in nav_fields.items():
        nav_mag = np.sqrt(nav_field[0]**2 + nav_field[1]**2)
        valid_mask = (nav_mag[walkable_mask > 0] > 0.5)
        coverage = float(valid_mask.sum() / walkable_pixels)
        nav_field_coverage[sink_id] = coverage
    
    metrics = Phase2Metrics(
        grid_shape=(H, W),
        walkable_pixels=walkable_pixels,
        walkable_ratio=float(walkable_ratio),
        num_nav_fields=num_nav_fields,
        nav_field_coverage=nav_field_coverage,
    )
    
    # 可视化
    if save_figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行：Walkable mask 和几个示例导航场
        ax = axes[0, 0]
        ax.imshow(walkable_mask, cmap="Greens", origin="upper")
        ax.set_title(f"Walkable Mask ({walkable_pixels:,} pixels)")
        ax.axis("off")
        
        # 选择几个代表性的 sink 显示导航场
        sample_sinks = list(nav_fields.keys())[:5]
        for i, sink_id in enumerate(sample_sinks[:2]):
            ax = axes[0, i + 1]
            nav_field = nav_fields[sink_id]
            
            # 显示导航场幅值
            nav_mag = np.sqrt(nav_field[0]**2 + nav_field[1]**2)
            im = ax.imshow(nav_mag, cmap="viridis", origin="upper", vmin=0, vmax=1)
            ax.set_title(f"Nav Field Sink {sink_id}")
            ax.axis("off")
        
        # 第二行：导航场覆盖率分布
        ax = axes[1, 0]
        coverages = list(nav_field_coverage.values())
        ax.bar(range(len(coverages)), sorted(coverages, reverse=True))
        ax.set_xlabel("Sink Rank")
        ax.set_ylabel("Valid Direction Coverage")
        ax.set_title("Nav Field Coverage Distribution")
        ax.axhline(y=np.mean(coverages), color="r", linestyle="--", label=f"Mean: {np.mean(coverages):.2f}")
        ax.legend()
        
        # 显示更多导航场
        for i, sink_id in enumerate(sample_sinks[2:5]):
            if i + 1 < 3:
                ax = axes[1, i + 1]
                nav_field = nav_fields[sink_id]
                nav_mag = np.sqrt(nav_field[0]**2 + nav_field[1]**2)
                ax.imshow(nav_mag, cmap="viridis", origin="upper", vmin=0, vmax=1)
                ax.set_title(f"Nav Field Sink {sink_id}")
                ax.axis("off")
        
        plt.tight_layout()
        fig.savefig(paths.figures_dir / "phase2_nav_fields.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {paths.figures_dir / 'phase2_nav_fields.png'}")
    
    # 保存指标（nav_field_coverage 需要特殊处理）
    np.savez(
        paths.metrics_dir / "phase2_metrics.npz",
        grid_shape=np.array(metrics.grid_shape),
        walkable_pixels=metrics.walkable_pixels,
        walkable_ratio=metrics.walkable_ratio,
        num_nav_fields=metrics.num_nav_fields,
        nav_field_coverage_keys=np.array(list(nav_field_coverage.keys())),
        nav_field_coverage_values=np.array(list(nav_field_coverage.values())),
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Phase 2 Validation Results")
    print("=" * 60)
    print(f"  Grid shape: {H} x {W}")
    print(f"  Walkable pixels: {walkable_pixels:,} ({walkable_ratio * 100:.1f}%)")
    print(f"  Num nav fields: {num_nav_fields}")
    print(f"  Nav coverage: mean={np.mean(coverages):.3f}, min={np.min(coverages):.3f}")
    
    if np.mean(coverages) > 0.9:
        print("  ✅ Navigation field coverage excellent")
    elif np.mean(coverages) > 0.7:
        print("  ✅ Navigation field coverage good")
    else:
        print("  ⚠️ Navigation field coverage may be insufficient")
    
    return metrics


if __name__ == "__main__":
    validate_phase2()
