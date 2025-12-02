"""
Phase 4 Diffusion Policy 可视化脚本

生成用于展示的高质量图表：
1. 全局轨迹图（密集）
2. 局部放大图（展示详细移动过程）
3. 统计分析图
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# 导入 Nav-Guided Policy
from nav_guided_inference import NavGuidedDiffusionPolicy, run_simulation


def create_visualization(
    trajectories: list,
    walkable_mask: np.ndarray,
    distance_field: np.ndarray,
    nav_field: np.ndarray,
    init_dist: np.ndarray,
    final_dist: np.ndarray,
    output_dir: Path,
    guidance_scale: float = 2.5,
):
    """创建完整的可视化"""
    
    H, W = walkable_mask.shape
    
    # ========== 1. 全局轨迹图 ==========
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 背景：道路网络
    ax.imshow(walkable_mask.T, cmap="gray", origin="lower", alpha=0.4)
    
    # 绘制所有轨迹
    for traj in trajectories:
        # 根据轨迹长度设置颜色深浅
        n_points = len(traj)
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_points))
        
        # 使用 LineCollection 绘制渐变轨迹
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=colors[:-1], linewidth=1.5, alpha=0.7)
        ax.add_collection(lc)
        
        # 起点（绿色）和终点（红色）
        ax.scatter(traj[0, 0], traj[0, 1], c='lime', s=20, zorder=5, edgecolors='darkgreen', linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=20, zorder=5, edgecolors='darkred', linewidths=0.5)
    
    # 统计信息
    dist_change = final_dist - init_dist
    approaching_rate = (dist_change < 0).mean() * 100
    
    ax.set_title(f"Diffusion Policy Trajectories (n={len(trajectories)})\n"
                 f"Approaching Rate: {approaching_rate:.1f}%  |  "
                 f"Mean Distance Change: {dist_change.mean():.1f} px", fontsize=14)
    ax.set_xlabel("Y (pixels)", fontsize=12)
    ax.set_ylabel("X (pixels)", fontsize=12)
    ax.set_aspect("equal")
    
    # 图例
    legend_elements = [
        mpatches.Patch(color='lime', label='Start'),
        mpatches.Patch(color='red', label='End'),
        plt.Line2D([0], [0], color='steelblue', linewidth=2, label='Trajectory')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_dir / "phase4_global_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'phase4_global_trajectories.png'}")
    
    # ========== 2. 局部放大图 ==========
    # 选择几条有代表性的轨迹进行放大展示
    # 找到最长的几条轨迹
    traj_lengths = [len(t) for t in trajectories]
    sorted_indices = np.argsort(traj_lengths)[::-1]
    
    # 选择 4 条轨迹做局部放大
    selected_indices = sorted_indices[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for ax_idx, traj_idx in enumerate(selected_indices):
        ax = axes[ax_idx]
        traj = trajectories[traj_idx]
        
        # 计算边界框（带 padding）
        padding = 50
        y_min, y_max = traj[:, 0].min() - padding, traj[:, 0].max() + padding
        x_min, x_max = traj[:, 1].min() - padding, traj[:, 1].max() + padding
        
        # 确保在地图范围内
        y_min, y_max = max(0, y_min), min(H, y_max)
        x_min, x_max = max(0, x_min), min(W, x_max)
        
        # 绘制局部道路
        local_mask = walkable_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
        ax.imshow(local_mask.T, cmap="gray", origin="lower", alpha=0.5,
                  extent=[y_min, y_max, x_min, x_max])
        
        # 绘制导航场（箭头）
        step = 15
        y_range = np.arange(int(y_min), int(y_max), step)
        x_range = np.arange(int(x_min), int(x_max), step)
        Y, X = np.meshgrid(y_range, x_range, indexing='ij')
        
        U = np.zeros_like(Y, dtype=float)
        V = np.zeros_like(X, dtype=float)
        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                if 0 <= y < H and 0 <= x < W and walkable_mask[y, x]:
                    U[i, j] = nav_field[0, y, x]
                    V[i, j] = nav_field[1, y, x]
        
        ax.quiver(Y, X, U, V, color='lightblue', alpha=0.5, scale=25, width=0.003)
        
        # 绘制轨迹（带时间标记）
        n_points = len(traj)
        colors = plt.cm.plasma(np.linspace(0, 1, n_points))
        
        for i in range(n_points - 1):
            ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=colors[i], linewidth=2.5, alpha=0.9)
        
        # 每隔一定步数标记时间点
        time_step = max(1, n_points // 10)
        for i in range(0, n_points, time_step):
            ax.scatter(traj[i, 0], traj[i, 1], c=[colors[i]], s=40, zorder=6, edgecolors='white', linewidths=0.5)
            ax.annotate(f't={i}', (traj[i, 0], traj[i, 1]), fontsize=8, 
                       xytext=(3, 3), textcoords='offset points')
        
        # 起点和终点
        ax.scatter(traj[0, 0], traj[0, 1], c='lime', s=100, zorder=7, 
                  edgecolors='darkgreen', linewidths=2, marker='o', label='Start')
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, zorder=7,
                  edgecolors='darkred', linewidths=2, marker='X', label='End')
        
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(x_min, x_max)
        ax.set_aspect("equal")
        ax.set_title(f"Trajectory {traj_idx+1}: {n_points} steps\n"
                    f"Distance: {init_dist[traj_idx]:.0f} → {final_dist[traj_idx]:.0f} px", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle("Detailed Trajectory Analysis (Local View)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "phase4_local_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'phase4_local_trajectories.png'}")
    
    # ========== 3. 统计分析图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 3.1 距离变化分布
    ax = axes[0, 0]
    ax.hist(dist_change, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(x=dist_change.mean(), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean: {dist_change.mean():.1f}')
    ax.set_xlabel("Distance Change (pixels)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Distance Change to Sink", fontsize=12)
    ax.legend()
    
    # 3.2 初始 vs 最终距离
    ax = axes[0, 1]
    ax.scatter(init_dist, final_dist, c='steelblue', alpha=0.6, s=30)
    max_dist = max(init_dist.max(), final_dist.max())
    ax.plot([0, max_dist], [0, max_dist], 'r--', linewidth=2, label='No change line')
    ax.set_xlabel("Initial Distance (pixels)", fontsize=11)
    ax.set_ylabel("Final Distance (pixels)", fontsize=11)
    ax.set_title("Initial vs Final Distance to Sink", fontsize=12)
    ax.legend()
    ax.set_aspect('equal')
    
    # 3.3 轨迹长度分布
    ax = axes[1, 0]
    traj_lengths = [len(t) for t in trajectories]
    ax.hist(traj_lengths, bins=20, color='forestgreen', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(traj_lengths), color='orange', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(traj_lengths):.0f}')
    ax.set_xlabel("Trajectory Length (steps)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Trajectory Lengths", fontsize=12)
    ax.legend()
    
    # 3.4 到达率统计
    ax = axes[1, 1]
    reached = final_dist < 10
    approaching = dist_change < 0
    
    categories = ['Reached\n(dist<10)', 'Approaching\n(dist↓)', 'Not Approaching\n(dist↑)']
    counts = [reached.sum(), (~reached & approaching).sum(), (~approaching).sum()]
    colors = ['forestgreen', 'steelblue', 'coral']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Agent Behavior Summary", fontsize=12)
    
    # 添加百分比标签
    total = len(trajectories)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{count/total*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle(f"Phase 4: Diffusion Policy Statistics (n={len(trajectories)})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "phase4_statistics.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'phase4_statistics.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    # 路径
    if args.checkpoint is None:
        args.checkpoint = PROJECT_ROOT / "data" / "output" / "phase4_checkpoints" / "best.pt"
    else:
        args.checkpoint = Path(args.checkpoint)
    
    if args.output is None:
        args.output = PROJECT_ROOT / "data" / "output" / "figures"
    else:
        args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Phase 4 Visualization")
    print("="*60)
    
    # 加载数据
    print("Loading data...")
    nav_data = np.load(PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz")
    nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)
    walkable_mask = np.load(PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy")
    distance_field = np.load(PROJECT_ROOT / "data" / "processed" / "distance_field.npy")
    
    H, W = walkable_mask.shape
    print(f"Map size: {H} x {W}")
    
    # 选择起点（分布更均匀）
    np.random.seed(123)
    walkable_points = np.argwhere(walkable_mask)
    nav_mag = np.sqrt(nav_field[0]**2 + nav_field[1]**2)
    
    # 选择有效导航场的点
    point_mags = nav_mag[walkable_points[:, 0], walkable_points[:, 1]]
    good_indices = np.where(point_mags > 0.3)[0]
    
    # 尽量均匀分布
    selected = np.random.choice(good_indices, min(args.num_agents, len(good_indices)), replace=False)
    start_positions = walkable_points[selected].astype(np.float32)
    
    print(f"Selected {len(start_positions)} start positions")
    
    # 创建 Policy
    print("Loading model...")
    policy = NavGuidedDiffusionPolicy(args.checkpoint, guidance_scale=args.guidance)
    
    # 运行仿真
    print(f"Running simulation (n={len(start_positions)}, max_steps={args.max_steps})...")
    trajectories, init_dist, final_dist = run_simulation(
        policy, nav_field, walkable_mask, distance_field,
        start_positions, max_steps=args.max_steps
    )
    
    # 生成可视化
    print("\nGenerating visualizations...")
    create_visualization(
        trajectories, walkable_mask, distance_field, nav_field,
        init_dist, final_dist, args.output, args.guidance
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
