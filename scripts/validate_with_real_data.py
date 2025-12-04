"""
验证框架: 基于真实经验数据的仿真验证

核心原则:
- Phase 3 和 Phase 4 应该用同样的真实数据进行验证
- 不应该只验证 Phase 4 是否接近 Phase 3
- 真实数据包括: Road Network, OD Flow, 通勤统计

已知 Limitations:
1. 当前 Phase 2 使用全局导航场（所有 sink 权重叠加）
   - 没有建模个体的目的地选择
   - 导致仿真的 OD 分布与真实 OD 不匹配
   
2. Phase 4 学习 Phase 3 的轨迹是自我循环
   - 学生不可能超过老师
   - 应该让两者都对照真实数据
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr


def validate_road_alignment(trajectories, walkable_mask):
    """验证轨迹是否在道路网络上
    
    Args:
        trajectories: (T, N, 2) 轨迹数据 (y, x) - 时间×agent×坐标
        walkable_mask: (H, W) 道路 mask
    
    Returns:
        on_road_rate: 在道路上的点的比例
    """
    H, W = walkable_mask.shape
    T, N, _ = trajectories.shape
    
    on_road = 0
    total = 0
    
    # 采样检查 (全量太慢)
    sample_steps = min(T, 1000)
    step_indices = np.linspace(0, T-1, sample_steps, dtype=int)
    
    for t in step_indices:
        for i in range(N):
            y = int(np.clip(trajectories[t, i, 0], 0, H-1))
            x = int(np.clip(trajectories[t, i, 1], 0, W-1))
            if walkable_mask[y, x] > 0:
                on_road += 1
            total += 1
    
    return on_road / total


def validate_od_distribution(trajectories, sinks_df, walkable_mask):
    """验证目的地分布是否匹配真实 OD
    
    Args:
        trajectories: (T, N, 2) 轨迹数据
    """
    T, N, _ = trajectories.shape
    H, W = walkable_mask.shape
    
    # 终点 = 最后一个时间步所有 agent 的位置
    endpoints = trajectories[-1, :, :]  # (N, 2)
    
    # 统计终点到 distance field 的值
    dist_field = np.load('data/processed/distance_field.npy')
    
    endpoint_distances = []
    for i in range(N):
        y = int(np.clip(endpoints[i, 0], 0, H-1))
        x = int(np.clip(endpoints[i, 1], 0, W-1))
        endpoint_distances.append(dist_field[y, x])
    
    return np.array(endpoint_distances)


def validate_travel_distance(trajectories, resolution_m=100):
    """验证行程距离分布是否合理
    
    Args:
        trajectories: (T, N, 2) - 时间×agent×坐标
        resolution_m: 每像素代表的米数
    """
    T, N, _ = trajectories.shape
    
    # 计算每个 agent 的总行程距离
    total_distances = []
    for i in range(N):
        agent_traj = trajectories[:, i, :]  # (T, 2)
        dists = np.sqrt(np.sum(np.diff(agent_traj, axis=0)**2, axis=1))
        total_dist = dists.sum() * resolution_m / 1000  # km
        total_distances.append(total_dist)
    
    return np.array(total_distances)


def main():
    print("=" * 70)
    print("基于真实数据的仿真验证")
    print("=" * 70)
    
    # 加载数据
    with h5py.File('data/output/trajectories.h5', 'r') as f:
        trajectories = f['positions'][:]
    
    mask = np.load('data/processed/walkable_mask.npy')
    sinks = pd.read_csv('data/processed/sinks_phase1.csv')
    
    N, T, _ = trajectories.shape
    print(f"\n轨迹: {N} 条, 每条 {T} 步")
    
    # 1. Road Alignment
    print("\n" + "=" * 50)
    print("1. Road Alignment (道路网络重合度)")
    print("=" * 50)
    
    on_road_rate = validate_road_alignment(trajectories, mask)
    print(f"在道路上的比例: {on_road_rate*100:.2f}%")
    
    if on_road_rate > 0.95:
        print("✓ 优秀: 轨迹几乎完全沿道路行驶")
    elif on_road_rate > 0.80:
        print("○ 良好: 大部分轨迹在道路上")
    else:
        print("✗ 需改进: 有较多轨迹偏离道路")
    
    # 2. 到达终点的距离
    print("\n" + "=" * 50)
    print("2. Destination Proximity (终点接近度)")
    print("=" * 50)
    
    endpoint_dists = validate_od_distribution(trajectories, sinks, mask)
    print(f"终点到 sink 的距离 (pixel):")
    print(f"  Mean: {endpoint_dists.mean():.1f}")
    print(f"  Median: {np.median(endpoint_dists):.1f}")
    print(f"  < 50 px (5km): {(endpoint_dists < 50).mean()*100:.1f}%")
    print(f"  < 100 px (10km): {(endpoint_dists < 100).mean()*100:.1f}%")
    
    # 3. 行程距离
    print("\n" + "=" * 50)
    print("3. Travel Distance (行程距离分布)")
    print("=" * 50)
    
    distances_km = validate_travel_distance(trajectories)
    print(f"行程距离 (km):")
    print(f"  Mean: {distances_km.mean():.1f}")
    print(f"  Median: {np.median(distances_km):.1f}")
    print(f"  Std: {distances_km.std():.1f}")
    print(f"  Range: [{distances_km.min():.1f}, {distances_km.max():.1f}]")
    
    # 典型通勤距离参考 (US average: ~15 miles ≈ 24 km)
    reasonable_range = (5, 50)  # km
    in_range = ((distances_km >= reasonable_range[0]) & 
                (distances_km <= reasonable_range[1])).mean()
    print(f"  在合理范围 [{reasonable_range[0]}-{reasonable_range[1]} km]: {in_range*100:.1f}%")
    
    # 4. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 4.1 终点距离分布
    ax = axes[0]
    ax.hist(endpoint_dists * 0.1, bins=50, alpha=0.7, color='steelblue')  # 转换为 km
    ax.axvline(endpoint_dists.mean() * 0.1, color='red', linestyle='--', label='Mean')
    ax.set_xlabel('Distance to Sink (km)')
    ax.set_ylabel('Count')
    ax.set_title('Endpoint Distance to Nearest Sink')
    ax.legend()
    
    # 4.2 行程距离分布
    ax = axes[1]
    ax.hist(distances_km, bins=50, alpha=0.7, color='coral')
    ax.axvline(distances_km.mean(), color='red', linestyle='--', label='Mean')
    ax.axvline(24, color='green', linestyle=':', label='US Avg (24km)')
    ax.set_xlabel('Travel Distance (km)')
    ax.set_ylabel('Count')
    ax.set_title('Travel Distance Distribution')
    ax.legend()
    
    # 4.3 汇总指标
    ax = axes[2]
    metrics = {
        'Road\nAlignment': on_road_rate,
        'Reached\n(<10km)': (endpoint_dists < 100).mean(),
        'Reasonable\nDistance': in_range,
    }
    bars = ax.bar(metrics.keys(), metrics.values(), color=['steelblue', 'coral', 'green'], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Rate')
    ax.set_title('Validation Metrics Summary')
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    output_path = 'data/output/figures/validation_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表保存到: {output_path}")
    plt.close()
    
    # 5. 总结
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    print(f"""
当前仿真质量:
  ✓ Road Alignment: {on_road_rate*100:.1f}% (轨迹在道路上)
  △ Destination: {(endpoint_dists < 100).mean()*100:.1f}% 到达 10km 内
  ○ Distance: {in_range*100:.1f}% 在合理通勤范围

已知 Limitations:
  1. 全局导航场: 所有 agent 共享混合目的地，与真实 OD 分布不匹配
  2. 固定步数仿真: 部分轨迹未到达目的地
  3. Phase 4 学习 Phase 3: 无外部真相验证

改进方向:
  1. 为每个 agent 采样独立的目的地 (基于 OD 概率)
  2. 使用到达终止条件，而非固定步数
  3. 引入真实 GPS 轨迹数据作为学习目标
""")


if __name__ == "__main__":
    main()
