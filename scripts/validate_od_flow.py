"""
方案 B: OD 流量验证
- 从仿真轨迹统计 O-D 流量分布
- 与真实 OD 数据对比
- 计算相关系数

验证问题: 仿真是否还原了宏观 OD 流量分布？
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_sinks():
    """加载 sink 位置 (lat, lon)"""
    sinks = pd.read_csv('data/processed/sinks_phase1.csv')
    return sinks[['lat', 'lon', 'total_flow']].values


def load_trajectories():
    """加载仿真轨迹的起点和终点"""
    with h5py.File('data/output/trajectories.h5', 'r') as f:
        # 轨迹数据: (N_traj, T, 2) 其中 2 是 (y, x) in pixels
        positions = f['positions'][:]
        
    N, T, _ = positions.shape
    print(f"Loaded {N} trajectories, {T} steps each")
    
    # 起点和终点 (pixel coords)
    origins = positions[:, 0, :]      # (N, 2)
    destinations = positions[:, -1, :] # (N, 2)
    
    return origins, destinations


def pixel_to_latlon(pixel_coords):
    """将 pixel 坐标转换为 lat/lon (近似)
    
    需要知道地图的边界和分辨率
    这里使用 Phase 1 的投影参数
    """
    # 从 Phase 1 config 获取投影参数
    from phase1 import config as p1_config
    
    # pixel (y, x) -> (lat, lon)
    # y 对应 lat (南北), x 对应 lon (东西)
    y = pixel_coords[:, 0]
    x = pixel_coords[:, 1]
    
    # 近似转换 (假设 100m/pixel)
    # 需要知道地图原点
    # 这里简化处理，使用 sinks 的 lat/lon 范围来推断
    sinks = pd.read_csv('data/processed/sinks_phase1.csv')
    lat_range = (sinks['lat'].min(), sinks['lat'].max())
    lon_range = (sinks['lon'].min(), sinks['lon'].max())
    
    # 加载地图尺寸
    mask = np.load('data/processed/walkable_mask.npy')
    H, W = mask.shape
    
    # 线性映射
    lat = lat_range[0] + (lat_range[1] - lat_range[0]) * (1 - y / H)  # y 从上到下，lat 从北到南
    lon = lon_range[0] + (lon_range[1] - lon_range[0]) * (x / W)
    
    return np.stack([lat, lon], axis=1)


def assign_to_sink(positions_latlon, sinks):
    """将每个位置分配到最近的 sink
    
    Args:
        positions_latlon: (N, 2) lat/lon
        sinks: (M, 3) [lat, lon, total_flow]
    
    Returns:
        sink_ids: (N,) 最近 sink 的索引
    """
    sink_coords = sinks[:, :2]  # (M, 2)
    
    # 计算距离矩阵
    dists = cdist(positions_latlon, sink_coords, metric='euclidean')
    
    # 分配到最近的 sink
    sink_ids = np.argmin(dists, axis=1)
    
    return sink_ids


def compute_simulated_od_matrix(origin_sinks, dest_sinks, n_sinks):
    """统计仿真的 OD 矩阵"""
    od_matrix = np.zeros((n_sinks, n_sinks), dtype=np.int32)
    
    for o, d in zip(origin_sinks, dest_sinks):
        od_matrix[o, d] += 1
    
    return od_matrix


def load_real_od_matrix(sinks_df):
    """从真实 OD 数据构建 OD 矩阵
    
    注意: 这需要将 tract 级别的 OD 聚合到 sink 级别
    简化处理: 使用 sink 的 total_flow 作为目的地吸引力
    """
    # 这里简化: 假设 OD 分布正比于 sink 的 flow 权重
    # 真实做法需要将 tract centroid 映射到 sink
    
    n_sinks = len(sinks_df)
    flows = sinks_df['total_flow'].values
    
    # 简化假设: P(O->D) ∝ flow(D)
    # 即目的地的吸引力正比于其 total_flow
    flow_probs = flows / flows.sum()
    
    return flow_probs


def compute_target_sink_from_trajectory(positions_px, sinks):
    """根据轨迹移动方向判断目标 sink
    
    方法: 看轨迹整体移动方向，与哪个 sink 的方向最一致
    """
    # 起点和终点
    origins = positions_px[:, 0, :]   # (N, 2)
    endpoints = positions_px[:, -1, :] # (N, 2)
    
    # 轨迹整体位移方向
    displacements = endpoints - origins  # (N, 2)
    
    # 转换 sink 到 pixel 坐标
    sinks_ll = sinks[:, :2]  # (M, 2) lat/lon
    
    # 需要将 sink 的 lat/lon 转为 pixel
    sinks_df = pd.read_csv('data/processed/sinks_phase1.csv')
    lat_range = (sinks_df['lat'].min(), sinks_df['lat'].max())
    lon_range = (sinks_df['lon'].min(), sinks_df['lon'].max())
    
    mask = np.load('data/processed/walkable_mask.npy')
    H, W = mask.shape
    
    # lat/lon -> pixel (y, x)
    sinks_y = (1 - (sinks_ll[:, 0] - lat_range[0]) / (lat_range[1] - lat_range[0])) * H
    sinks_x = (sinks_ll[:, 1] - lon_range[0]) / (lon_range[1] - lon_range[0]) * W
    sinks_px = np.stack([sinks_y, sinks_x], axis=1)  # (M, 2)
    
    # 对每条轨迹，计算从起点到各个 sink 的方向
    N = len(origins)
    M = len(sinks_px)
    target_sinks = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        origin = origins[i]  # (2,)
        disp = displacements[i]  # (2,)
        
        disp_mag = np.linalg.norm(disp)
        if disp_mag < 1:  # 几乎没移动
            target_sinks[i] = -1
            continue
        
        disp_dir = disp / disp_mag  # 归一化方向
        
        # 从起点到各个 sink 的方向
        to_sinks = sinks_px - origin  # (M, 2)
        to_sinks_mag = np.linalg.norm(to_sinks, axis=1)
        to_sinks_mag[to_sinks_mag < 1] = 1  # 避免除零
        to_sinks_dir = to_sinks / to_sinks_mag[:, None]
        
        # 计算余弦相似度
        cos_sims = np.dot(to_sinks_dir, disp_dir)
        
        # 选择方向最一致的 sink
        target_sinks[i] = np.argmax(cos_sims)
    
    return target_sinks


def main():
    print("=" * 60)
    print("OD 流量验证 (基于移动方向)")
    print("=" * 60)
    
    # 1. 加载数据
    sinks = load_sinks()
    n_sinks = len(sinks)
    print(f"\nSinks: {n_sinks}")
    
    # 加载完整轨迹
    with h5py.File('data/output/trajectories.h5', 'r') as f:
        positions_px = f['positions'][:]
    n_traj = len(positions_px)
    print(f"Trajectories: {n_traj}")
    
    origins_px = positions_px[:, 0, :]
    
    # 2. 根据移动方向判断目标 sink
    print("\n分析轨迹移动方向...")
    target_sinks = compute_target_sink_from_trajectory(positions_px, sinks)
    
    # 过滤无效轨迹
    valid_mask = target_sinks >= 0
    print(f"有效轨迹: {valid_mask.sum()} / {n_traj}")
    
    # 3. 统计目的地分布 (基于移动方向)
    sim_dest_dist = np.zeros(n_sinks)
    for sink_id in target_sinks[valid_mask]:
        sim_dest_dist[sink_id] += 1
    sim_dest_dist = sim_dest_dist / sim_dest_dist.sum()
    
    # 4. 真实目的地分布 (基于 sink 的 total_flow)
    sinks_df = pd.read_csv('data/processed/sinks_phase1.csv')
    real_dest_dist = sinks_df['total_flow'].values
    real_dest_dist = real_dest_dist / real_dest_dist.sum()
    
    # 5. 计算相关性
    pearson_r, pearson_p = pearsonr(sim_dest_dist, real_dest_dist)
    spearman_r, spearman_p = spearmanr(sim_dest_dist, real_dest_dist)
    
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"Pearson 相关系数: {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"Spearman 相关系数: {spearman_r:.4f} (p={spearman_p:.4e})")
    
    # 6. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 6.1 目的地分布对比
    ax = axes[0]
    x = np.arange(n_sinks)
    width = 0.35
    ax.bar(x - width/2, real_dest_dist, width, label='Real (OD Flow)', alpha=0.7, color='steelblue')
    ax.bar(x + width/2, sim_dest_dist, width, label='Simulated', alpha=0.7, color='coral')
    ax.set_xlabel('Sink ID')
    ax.set_ylabel('Probability')
    ax.set_title('Destination Distribution: Real vs Simulated')
    ax.legend()
    
    # 6.2 散点图
    ax = axes[1]
    ax.scatter(real_dest_dist, sim_dest_dist, alpha=0.7, s=80, c='steelblue')
    max_val = max(real_dest_dist.max(), sim_dest_dist.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x', alpha=0.7)
    ax.set_xlabel('Real Distribution')
    ax.set_ylabel('Simulated Distribution')
    ax.set_title(f'Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}')
    ax.legend()
    
    plt.tight_layout()
    
    output_path = 'data/output/figures/od_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表保存到: {output_path}")
    plt.close()
    
    # 7. 详细统计
    print("\n" + "=" * 60)
    print("详细统计")
    print("=" * 60)
    
    # Top 5 目的地
    print("\nTop 5 目的地 (Real vs Simulated):")
    top_real = np.argsort(real_dest_dist)[::-1][:5]
    for i, sink_id in enumerate(top_real):
        print(f"  {i+1}. Sink {sink_id}: Real={real_dest_dist[sink_id]:.3f}, Sim={sim_dest_dist[sink_id]:.3f}")
    
    return pearson_r, spearman_r


if __name__ == "__main__":
    main()
