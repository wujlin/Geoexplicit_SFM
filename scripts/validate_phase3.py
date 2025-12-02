"""
Phase 3 轨迹仿真 验证脚本

验证内容：
1. 轨迹数据完整性
2. 道路遵循率 (位置是否在道路上)
3. 速度分布 (零速度比例)
4. 方向一致性 (velocity vs nav)
5. 距离递减率 (是否朝向 Sink)
"""

from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_phase3(output_dir: Path = None, sample_ratio: float = 0.01):
    """
    运行 Phase 3 验证
    
    Args:
        output_dir: 输出目录
        sample_ratio: 采样比例 (默认 1%)
    """
    print(f"\n{'='*60}")
    print("Phase 3: 轨迹仿真 验证")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. 加载数据
    traj_path = PROJECT_ROOT / "data" / "output" / "trajectories.h5"
    nav_path = PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz"
    dist_path = PROJECT_ROOT / "data" / "processed" / "distance_field.npy"
    mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
    
    if not traj_path.exists():
        print(f"  错误: 轨迹文件不存在: {traj_path}")
        return None
    
    with h5py.File(traj_path, "r") as f:
        positions = f["positions"][:]  # (T, N, 2)
        velocities = f["velocities"][:]  # (T, N, 2)
    
    T, N, _ = positions.shape
    
    nav_data = np.load(nav_path)
    nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)
    distance_field = np.load(dist_path)
    walkable_mask = np.load(mask_path)
    H, W = walkable_mask.shape
    
    print(f"\n[1] 数据规模")
    print(f"    时间步: {T}")
    print(f"    智能体数: {N}")
    print(f"    总帧数: {T * N:,}")
    
    results["T"] = T
    results["N"] = N
    results["total_frames"] = T * N
    
    # 2. 采样分析 (全量太慢)
    num_samples = int(T * N * sample_ratio)
    print(f"\n[2] 采样分析 (n={num_samples:,}, {sample_ratio*100:.1f}%)")
    
    np.random.seed(42)
    sample_t = np.random.randint(1, T, num_samples)  # 从 t=1 开始，方便计算距离变化
    sample_n = np.random.randint(0, N, num_samples)
    
    # 收集指标
    on_road = []
    speeds = []
    cos_vel_nav = []
    dist_decrease = []
    
    for t, n in tqdm(zip(sample_t, sample_n), total=num_samples, desc="    分析中"):
        pos = positions[t, n]
        vel = velocities[t, n]
        prev_pos = positions[t-1, n]
        
        y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
        prev_y, prev_x = int(np.clip(prev_pos[0], 0, H-1)), int(np.clip(prev_pos[1], 0, W-1))
        
        # 道路检查
        on_road.append(walkable_mask[y, x] > 0)
        
        # 速度
        speed = np.linalg.norm(vel)
        speeds.append(speed)
        
        # velocity vs nav
        nav = nav_field[:, y, x]
        nav_norm = np.linalg.norm(nav)
        if speed > 0.01 and nav_norm > 0.01:
            cos = np.dot(vel / speed, nav / nav_norm)
            cos_vel_nav.append(cos)
        
        # 距离变化
        curr_dist = distance_field[y, x]
        prev_dist = distance_field[prev_y, prev_x]
        if prev_dist > 0:  # 排除已在 Sink 的情况
            dist_decrease.append(curr_dist < prev_dist)
    
    on_road = np.array(on_road)
    speeds = np.array(speeds)
    cos_vel_nav = np.array(cos_vel_nav)
    dist_decrease = np.array(dist_decrease)
    
    # 3. 统计结果
    print(f"\n[3] 道路遵循率")
    on_road_rate = on_road.mean()
    print(f"    在道路上: {on_road_rate*100:.4f}%")
    results["on_road_rate"] = float(on_road_rate)
    
    print(f"\n[4] 速度分布")
    zero_speed_rate = (speeds < 0.01).mean()
    print(f"    平均速度: {speeds.mean():.4f} px/step")
    print(f"    速度标准差: {speeds.std():.4f}")
    print(f"    零速度帧 (<0.01): {zero_speed_rate*100:.4f}%")
    print(f"    最大速度: {speeds.max():.4f}")
    
    results["speed_mean"] = float(speeds.mean())
    results["speed_std"] = float(speeds.std())
    results["zero_speed_rate"] = float(zero_speed_rate)
    results["speed_max"] = float(speeds.max())
    
    print(f"\n[5] 方向一致性 (velocity vs nav)")
    cos_mean = cos_vel_nav.mean()
    positive_rate = (cos_vel_nav > 0).mean()
    print(f"    cos_sim mean: {cos_mean:.4f}")
    print(f"    正向 (>0): {positive_rate*100:.2f}%")
    print(f"    强正向 (>0.5): {(cos_vel_nav > 0.5).mean()*100:.2f}%")
    
    results["cos_vel_nav_mean"] = float(cos_mean)
    results["cos_vel_nav_positive_rate"] = float(positive_rate)
    
    print(f"\n[6] 距离递减率 (朝向 Sink)")
    decrease_rate = dist_decrease.mean()
    print(f"    距离递减: {decrease_rate*100:.2f}%")
    
    results["distance_decrease_rate"] = float(decrease_rate)
    
    # 4. 可视化
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 速度分布
        ax = axes[0, 0]
        ax.hist(speeds, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(x=speeds.mean(), color="red", linestyle="--", 
                   label=f"Mean: {speeds.mean():.3f}")
        ax.set_xlabel("Speed (px/step)")
        ax.set_ylabel("Count")
        ax.set_title("Speed Distribution")
        ax.legend()
        
        # cos(vel, nav) 分布
        ax = axes[0, 1]
        ax.hist(cos_vel_nav, bins=50, color="green", alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=cos_vel_nav.mean(), color="red", linestyle="--",
                   label=f"Mean: {cos_vel_nav.mean():.3f}")
        ax.set_xlabel("cos(velocity, nav)")
        ax.set_ylabel("Count")
        ax.set_title("Velocity vs Nav Direction Alignment")
        ax.legend()
        
        # 轨迹可视化 (抽样)
        ax = axes[1, 0]
        ax.imshow(walkable_mask.T, cmap="gray", origin="lower", alpha=0.5)
        
        # 随机选几条轨迹
        np.random.seed(123)
        sample_agents = np.random.choice(N, min(50, N), replace=False)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sample_agents)))
        
        for agent, color in zip(sample_agents, colors):
            traj = positions[:, agent, :]
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.5, alpha=0.7)
        
        ax.set_title(f"Sample Trajectories (n={len(sample_agents)})")
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_aspect("equal")
        
        # 汇总指标
        ax = axes[1, 1]
        ax.axis("off")
        
        summary_text = f"""
Phase 3 Validation Summary
{'='*40}

Data Scale:
  • Time steps: {T:,}
  • Agents: {N:,}
  • Total frames: {T*N:,}

Road Following:
  • On-road rate: {on_road_rate*100:.4f}%

Speed:
  • Mean: {speeds.mean():.4f} px/step
  • Zero-speed rate: {zero_speed_rate*100:.4f}%

Direction Alignment (vel vs nav):
  • cos_sim mean: {cos_mean:.4f}
  • Positive rate: {positive_rate*100:.2f}%

Distance to Sink:
  • Decreasing rate: {decrease_rate*100:.2f}%
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.tight_layout()
        fig_path = output_dir / "phase3_validation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  图表保存至: {fig_path}")
        
        # 保存 JSON
        json_path = output_dir / "phase3_validation.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  结果保存至: {json_path}")
    
    # 5. 总结
    print(f"\n{'='*60}")
    passed = (on_road_rate > 0.99 and 
              zero_speed_rate < 0.01 and 
              positive_rate > 0.7 and
              decrease_rate > 0.8)
    results["validation_passed"] = bool(passed)
    status = "✅ 通过" if passed else "❌ 失败"
    print(f"Phase 3 验证状态: {status}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "data" / "output" / "validation"
    validate_phase3(output_dir)
