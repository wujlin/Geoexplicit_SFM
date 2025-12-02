"""
Phase 1 Sink 识别 验证脚本

验证内容：
1. Sink 数量和流量统计
2. Sink 是否在道路上
3. 空间分布可视化
"""

from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_phase1(output_dir: Path = None):
    """运行 Phase 1 验证"""
    print(f"\n{'='*60}")
    print("Phase 1: Sink 识别 验证")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. 加载数据
    sinks_path = PROJECT_ROOT / "data" / "processed" / "sinks_phase1.csv"
    mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
    
    if not sinks_path.exists():
        print(f"  错误: Sink 文件不存在: {sinks_path}")
        return None
    
    sinks = pd.read_csv(sinks_path)
    walkable_mask = np.load(mask_path)
    H, W = walkable_mask.shape
    
    # 检查是否有 pixel 坐标，如果没有则需要从 lat/lon 转换
    if "pixel_y" not in sinks.columns:
        # 尝试从 data_loader 获取坐标转换参数
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from phase1.data_loader import DataLoader
            loader = DataLoader()
            bounds = loader.get_bounds()
            
            # 转换 lat/lon 到 pixel
            pixel_ys = []
            pixel_xs = []
            for _, row in sinks.iterrows():
                py, px = loader.latlon_to_pixel(row["lat"], row["lon"])
                pixel_ys.append(py)
                pixel_xs.append(px)
            sinks["pixel_y"] = pixel_ys
            sinks["pixel_x"] = pixel_xs
            print(f"  (已从 lat/lon 转换 pixel 坐标)")
        except Exception as e:
            print(f"  警告: 无法转换坐标 - {e}")
            # 使用简单的线性映射作为备选
            lat_min, lat_max = sinks["lat"].min(), sinks["lat"].max()
            lon_min, lon_max = sinks["lon"].min(), sinks["lon"].max()
            sinks["pixel_y"] = ((sinks["lat"] - lat_min) / (lat_max - lat_min) * (H - 1)).astype(int)
            sinks["pixel_x"] = ((sinks["lon"] - lon_min) / (lon_max - lon_min) * (W - 1)).astype(int)
            print(f"  (使用简单线性映射转换坐标)")
    
    print(f"\n[1] 基本统计")
    print(f"    Sink 数量: {len(sinks)}")
    print(f"    总流量: {sinks['total_flow'].sum():,.0f}")
    print(f"    最大流量: {sinks['total_flow'].max():,.0f}")
    print(f"    最小流量: {sinks['total_flow'].min():,.0f}")
    print(f"    平均流量: {sinks['total_flow'].mean():,.0f}")
    print(f"    中位数流量: {sinks['total_flow'].median():,.0f}")
    
    results["num_sinks"] = len(sinks)
    results["total_flow"] = int(sinks["total_flow"].sum())
    results["max_flow"] = int(sinks["total_flow"].max())
    results["min_flow"] = int(sinks["total_flow"].min())
    results["mean_flow"] = float(sinks["total_flow"].mean())
    
    # 2. 验证 Sink 在道路上
    print(f"\n[2] 道路约束验证")
    on_road_count = 0
    off_road_sinks = []
    
    for idx, row in sinks.iterrows():
        y, x = int(row["pixel_y"]), int(row["pixel_x"])
        if 0 <= y < H and 0 <= x < W:
            if walkable_mask[y, x] > 0:
                on_road_count += 1
            else:
                off_road_sinks.append((idx, y, x))
        else:
            off_road_sinks.append((idx, y, x))
    
    on_road_rate = on_road_count / len(sinks) if len(sinks) > 0 else 0
    print(f"    在道路上: {on_road_count}/{len(sinks)} ({on_road_rate*100:.1f}%)")
    
    if off_road_sinks:
        print(f"    警告: {len(off_road_sinks)} 个 Sink 不在道路上!")
        for idx, y, x in off_road_sinks[:5]:
            print(f"      Sink {idx}: ({y}, {x})")
    
    results["on_road_count"] = on_road_count
    results["on_road_rate"] = float(on_road_rate)
    results["validation_passed"] = on_road_rate == 1.0
    
    # 3. 流量分布分析
    print(f"\n[3] 流量分布")
    flow_sorted = sinks["total_flow"].sort_values(ascending=False)
    cumsum = flow_sorted.cumsum()
    total = flow_sorted.sum()
    
    # Pareto 分析
    top10_flow = flow_sorted.head(10).sum()
    top10_pct = top10_flow / total * 100
    print(f"    Top 10 Sink 占比: {top10_pct:.1f}%")
    
    # 找到 80% 流量需要多少 Sink
    threshold_80 = 0.8 * total
    n_for_80 = (cumsum <= threshold_80).sum() + 1
    print(f"    80% 流量需要: {n_for_80} 个 Sink ({n_for_80/len(sinks)*100:.1f}%)")
    
    results["top10_flow_pct"] = float(top10_pct)
    results["n_sinks_for_80pct"] = int(n_for_80)
    
    # 4. 可视化
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图: 空间分布
        ax = axes[0]
        ax.imshow(walkable_mask.T, cmap="gray", origin="lower", alpha=0.5)
        scatter = ax.scatter(
            sinks["pixel_y"], sinks["pixel_x"],
            c=np.log10(sinks["total_flow"] + 1),
            s=sinks["total_flow"] / sinks["total_flow"].max() * 200 + 10,
            cmap="Reds", alpha=0.7, edgecolors="black", linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label="log10(flow)")
        ax.set_title(f"Sink Spatial Distribution (n={len(sinks)})")
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_aspect("equal")
        
        # 右图: 流量分布
        ax = axes[1]
        ax.bar(range(len(flow_sorted)), flow_sorted.values, color="steelblue", alpha=0.7)
        ax.axhline(y=flow_sorted.mean(), color="red", linestyle="--", label=f"Mean: {flow_sorted.mean():.0f}")
        ax.set_xlabel("Sink Index (sorted)")
        ax.set_ylabel("Flow")
        ax.set_title("Flow Distribution (Pareto)")
        ax.legend()
        ax.set_yscale("log")
        
        plt.tight_layout()
        fig_path = output_dir / "phase1_validation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  图表保存至: {fig_path}")
        
        # 保存 JSON
        json_path = output_dir / "phase1_validation.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  结果保存至: {json_path}")
    
    # 5. 总结
    print(f"\n{'='*60}")
    status = "✅ 通过" if results["validation_passed"] else "❌ 失败"
    print(f"Phase 1 验证状态: {status}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "data" / "output" / "validation"
    validate_phase1(output_dir)
