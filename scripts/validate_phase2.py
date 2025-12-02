"""
Phase 2 导航场构建 验证脚本

验证内容：
1. 导航场有效性 (magnitude > 0.9)
2. 方向正确性 (沿 nav 方向走，distance 减小)
3. 距离场连续性
"""

from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_phase2(output_dir: Path = None, num_samples: int = 10000):
    """运行 Phase 2 验证"""
    print(f"\n{'='*60}")
    print("Phase 2: 导航场构建 验证")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. 加载数据
    nav_path = PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz"
    dist_path = PROJECT_ROOT / "data" / "processed" / "distance_field.npy"
    mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
    
    nav_data = np.load(nav_path)
    nav_y = nav_data["nav_y"]
    nav_x = nav_data["nav_x"]
    distance_field = np.load(dist_path)
    walkable_mask = np.load(mask_path)
    
    H, W = walkable_mask.shape
    nav_field = np.stack([nav_y, nav_x], axis=0)  # (2, H, W)
    
    print(f"\n[1] 基本信息")
    print(f"    地图尺寸: {H} × {W}")
    print(f"    道路像素: {(walkable_mask > 0).sum():,}")
    print(f"    道路比例: {(walkable_mask > 0).mean()*100:.2f}%")
    
    results["map_size"] = [H, W]
    results["road_pixels"] = int((walkable_mask > 0).sum())
    results["road_ratio"] = float((walkable_mask > 0).mean())
    
    # 2. 导航场有效性
    print(f"\n[2] 导航场有效性")
    nav_magnitude = np.sqrt(nav_y**2 + nav_x**2)
    
    # 只在道路区域计算
    road_mask = walkable_mask > 0
    road_nav_mag = nav_magnitude[road_mask]
    
    valid_nav = road_nav_mag > 0.9
    valid_rate = valid_nav.mean()
    
    print(f"    道路上 |nav| > 0.9: {valid_rate*100:.2f}%")
    print(f"    道路上 |nav| mean: {road_nav_mag.mean():.4f}")
    print(f"    道路上 |nav| min: {road_nav_mag.min():.4f}")
    
    results["nav_valid_rate"] = float(valid_rate)
    results["nav_magnitude_mean"] = float(road_nav_mag.mean())
    results["nav_magnitude_min"] = float(road_nav_mag.min())
    
    # 3. 方向正确性验证
    print(f"\n[3] 方向正确性验证 (沿 nav 方向走，distance 应减小)")
    
    # 采样道路点
    road_ys, road_xs = np.where(road_mask)
    np.random.seed(42)
    sample_idx = np.random.choice(len(road_ys), min(num_samples, len(road_ys)), replace=False)
    
    correct_count = 0
    total_valid = 0
    step_size = 1.0
    
    for idx in tqdm(sample_idx, desc="    验证方向"):
        y, x = road_ys[idx], road_xs[idx]
        
        nav = nav_field[:, y, x]
        nav_norm = np.linalg.norm(nav)
        if nav_norm < 0.1:
            continue
        
        nav = nav / nav_norm
        
        # 沿 nav 方向走一步
        new_y = y + nav[0] * step_size
        new_x = x + nav[1] * step_size
        
        # 边界检查
        new_yi = int(np.clip(new_y, 0, H-1))
        new_xi = int(np.clip(new_x, 0, W-1))
        
        # 检查新位置是否在道路上
        if not walkable_mask[new_yi, new_xi]:
            continue
        
        total_valid += 1
        
        # 检查距离是否减小
        old_dist = distance_field[y, x]
        new_dist = distance_field[new_yi, new_xi]
        
        if new_dist < old_dist:
            correct_count += 1
    
    correct_rate = correct_count / total_valid if total_valid > 0 else 0
    print(f"    有效样本: {total_valid}")
    print(f"    方向正确率: {correct_rate*100:.2f}%")
    
    results["direction_samples"] = total_valid
    results["direction_correct_rate"] = float(correct_rate)
    
    # 4. 距离场分析
    print(f"\n[4] 距离场分析")
    road_dist = distance_field[road_mask]
    
    print(f"    距离范围: [{road_dist.min():.1f}, {road_dist.max():.1f}] px")
    print(f"    平均距离: {road_dist.mean():.1f} px")
    print(f"    距离=0 (Sink): {(road_dist == 0).sum()} 点")
    
    results["distance_min"] = float(road_dist.min())
    results["distance_max"] = float(road_dist.max())
    results["distance_mean"] = float(road_dist.mean())
    results["sink_pixels"] = int((road_dist == 0).sum())
    
    # 5. 可视化
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 距离场
        ax = axes[0]
        im = ax.imshow(distance_field.T, cmap="viridis", origin="lower")
        ax.set_title("Distance Field")
        plt.colorbar(im, ax=ax, label="Distance to Sink (px)")
        
        # 导航场幅度
        ax = axes[1]
        im = ax.imshow(nav_magnitude.T, cmap="hot", origin="lower", vmin=0, vmax=1.1)
        ax.set_title("Navigation Field Magnitude")
        plt.colorbar(im, ax=ax, label="|nav|")
        
        # 导航场方向 (抽样)
        ax = axes[2]
        ax.imshow(walkable_mask.T, cmap="gray", origin="lower", alpha=0.3)
        
        # 抽样绘制箭头
        step = 30
        Y, X = np.mgrid[0:H:step, 0:W:step]
        U = nav_y[::step, ::step]
        V = nav_x[::step, ::step]
        M = nav_magnitude[::step, ::step]
        
        # 只绘制有效区域
        mask_sub = walkable_mask[::step, ::step] > 0
        ax.quiver(Y[mask_sub], X[mask_sub], U[mask_sub], V[mask_sub], 
                  M[mask_sub], cmap="coolwarm", scale=30, alpha=0.8)
        ax.set_title("Navigation Direction Field")
        ax.set_aspect("equal")
        
        plt.tight_layout()
        fig_path = output_dir / "phase2_validation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  图表保存至: {fig_path}")
        
        # 保存 JSON
        json_path = output_dir / "phase2_validation.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  结果保存至: {json_path}")
    
    # 6. 总结
    print(f"\n{'='*60}")
    passed = results["nav_valid_rate"] > 0.9 and results["direction_correct_rate"] > 0.95
    results["validation_passed"] = passed
    status = "✅ 通过" if passed else "❌ 失败"
    print(f"Phase 2 验证状态: {status}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "data" / "output" / "validation"
    validate_phase2(output_dir)
