"""
Phase 3 验证：轨迹生成与 OD 分布评估

注意：trajectories.h5 的数据格式：
- positions: [T, N_agents, 2] 位置 (时间步在前)
- velocities: [T, N_agents, 2] 速度
- destinations: [T, N_agents] 每个时间步的目的地 sink ID
  （当 agent 到达并重生时，destination 会变化）
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from . import PathConfig, Phase3Metrics, get_path_config


def extract_trips_correct(destinations: np.ndarray, positions: np.ndarray, pixel_to_sink: np.ndarray, H: int, W: int) -> list:
    """
    从目的地序列中正确提取所有 trips
    
    每个 trip 的 OD 应该是：
    - Origin: agent 出发时所在位置对应的 sink
    - Destination: 该时间步的 dest_sink
    
    destinations: [T, N] 形状
    positions: [T, N, 2] 形状
    pixel_to_sink: [H*W] 形状，pixel -> sink 映射
    
    Returns:
        list of (origin_sink, dest_sink) tuples
    """
    trips = []
    T, n_agents = destinations.shape
    
    for i in range(n_agents):
        dest_seq = destinations[:, i]  # 这个 agent 的目的地序列
        
        # 找到目的地变化的位置（即到达时刻）
        changes = np.where(np.diff(dest_seq) != 0)[0] + 1
        
        # 第一个 trip: 从 t=0 开始
        segment_starts = [0] + list(changes)
        
        for seg_start in segment_starts:
            if seg_start >= T - 1:
                continue
            
            # 这个 trip 的目的地
            dest_sink = int(dest_seq[seg_start])
            
            # 这个 trip 的起点位置
            pos_y = int(np.clip(positions[seg_start, i, 0], 0, H - 1))
            pos_x = int(np.clip(positions[seg_start, i, 1], 0, W - 1))
            flat_idx = pos_y * W + pos_x
            
            origin_sink = pixel_to_sink[flat_idx] if flat_idx < len(pixel_to_sink) else -1
            
            if origin_sink >= 0 and origin_sink != dest_sink:  # 只统计非对角线
                trips.append((int(origin_sink), dest_sink))
    
    return trips


def validate_phase3(paths: PathConfig = None, save_figure: bool = True) -> Phase3Metrics:
    """
    验证 Phase 3 轨迹生成质量
    """
    if paths is None:
        paths = get_path_config()
    
    paths.ensure_dirs()
    
    # 加载轨迹数据
    import h5py
    with h5py.File(paths.trajectories_h5, "r") as f:
        destinations = f["destinations"][:]  # [T, N_agents]
        positions = f["positions"][:]  # [T, N_agents, 2]
        velocities = f["velocities"][:]  # [T, N_agents, 2]
    
    max_T, n_agents, _ = positions.shape
    
    # 加载 pixel → sink 映射
    walkable_mask = np.load(paths.walkable_mask)
    H, W = walkable_mask.shape
    
    # 尝试加载预构建的 pixel_to_sink 映射
    pixel_to_sink_path = paths.base_dir / "data" / "processed" / "pixel_to_sink.npy"
    if pixel_to_sink_path.exists():
        pixel_to_sink = np.load(pixel_to_sink_path)
    else:
        # 如果不存在，构建一个简化版本（使用简单的方法）
        print("  Building pixel-to-sink mapping (this may take a moment)...")
        from scipy.spatial import cKDTree
        
        # 加载 sink 数据
        sink_df = pd.read_csv(paths.sinks_csv)
        sink_coords = sink_df[["px", "py"]].values if "px" in sink_df.columns else None
        
        if sink_coords is None:
            # 使用 tract_pixel_mapping 和 tract_sink_mapping
            tract_pixel = pd.read_csv(paths.base_dir / "data" / "processed" / "tract_pixel_mapping.csv")
            tract_sink = pd.read_csv(paths.base_dir / "data" / "processed" / "tract_sink_mapping.csv")
            merged = tract_pixel.merge(tract_sink[["GEOID", "sink_id"]], on="GEOID")
            sink_coords = merged[["px", "py"]].values
            sink_ids = merged["sink_id"].values
            
            tree = cKDTree(sink_coords)
            
            # 为每个像素找最近的 sink
            ys, xs = np.mgrid[0:H, 0:W]
            all_coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
            _, indices = tree.query(all_coords, k=1)
            pixel_to_sink = sink_ids[indices].reshape(H * W)
        else:
            # 直接使用 sink 坐标
            tree = cKDTree(sink_coords)
            ys, xs = np.mgrid[0:H, 0:W]
            all_coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
            _, pixel_to_sink = tree.query(all_coords, k=1)
            pixel_to_sink = pixel_to_sink.astype(np.int32)
        
        # 保存以便复用
        np.save(pixel_to_sink_path, pixel_to_sink)
        print(f"  Saved pixel-to-sink mapping to {pixel_to_sink_path}")
    
    # 从目的地变化中提取 trips
    print("  Extracting trips from trajectory data...")
    trips = extract_trips_correct(destinations, positions, pixel_to_sink, H, W)
    n_trips = len(trips)
    
    # 速度统计
    speed = np.sqrt(velocities[:, :, 0]**2 + velocities[:, :, 1]**2)
    valid_speed = speed[speed > 0.01]  # 过滤静止状态
    mean_speed = float(valid_speed.mean()) if len(valid_speed) > 0 else 0
    std_speed = float(valid_speed.std()) if len(valid_speed) > 0 else 0
    
    # 加载真实 OD 数据（已聚合到 sink 级别）
    real_od_df = pd.read_csv(paths.od_matrix)
    
    # 只比较非对角线（外部）流量，因为模拟不产生内部流量
    real_od_df = real_od_df[real_od_df["origin_sink"] != real_od_df["dest_sink"]]
    
    # 构建真实 OD 分布字典
    real_od_by_sink = {}
    for _, row in real_od_df.iterrows():
        key = (int(row["origin_sink"]), int(row["dest_sink"]))
        real_od_by_sink[key] = row["flow"]
    
    # 统计模拟的 OD 分布
    simulated_od_by_sink = {}
    for o, d in trips:
        key = (o, d)
        simulated_od_by_sink[key] = simulated_od_by_sink.get(key, 0) + 1
    
    # ======== 方法1：整体 OD 分布相关性 ========
    # 构建对比向量
    all_keys = set(real_od_by_sink.keys()) | set(simulated_od_by_sink.keys())
    real_vec = np.array([real_od_by_sink.get(k, 0) for k in all_keys])
    sim_vec = np.array([simulated_od_by_sink.get(k, 0) for k in all_keys])
    
    # 归一化为分布
    real_dist = real_vec / real_vec.sum() if real_vec.sum() > 0 else real_vec
    sim_dist = sim_vec / sim_vec.sum() if sim_vec.sum() > 0 else sim_vec
    
    # 计算整体相关性
    if len(real_dist) > 1 and real_dist.std() > 0 and sim_dist.std() > 0:
        pearson_r, _ = stats.pearsonr(real_dist, sim_dist)
        spearman_r, _ = stats.spearmanr(real_dist, sim_dist)
    else:
        pearson_r, spearman_r = 0.0, 0.0
    
    # ======== 方法2：条件分布相关性（给定 origin，目的地分布相似度）========
    # 这是更好的指标，因为模拟中的 origin 频率与真实数据可能不同
    origin_sinks = set(k[0] for k in real_od_by_sink.keys())
    conditional_correlations = []
    
    for origin in origin_sinks:
        # 真实分布
        real_subset = {d: f for (o, d), f in real_od_by_sink.items() if o == origin}
        sim_subset = {d: f for (o, d), f in simulated_od_by_sink.items() if o == origin}
        
        if len(sim_subset) < 2:
            continue
        
        # 对齐目的地
        all_dests = set(real_subset.keys()) | set(sim_subset.keys())
        real_vals = np.array([real_subset.get(d, 0) for d in all_dests])
        sim_vals = np.array([sim_subset.get(d, 0) for d in all_dests])
        
        # 归一化
        real_vals = real_vals / real_vals.sum() if real_vals.sum() > 0 else real_vals
        sim_vals = sim_vals / sim_vals.sum() if sim_vals.sum() > 0 else sim_vals
        
        if real_vals.std() > 0 and sim_vals.std() > 0:
            corr, _ = stats.pearsonr(real_vals, sim_vals)
            conditional_correlations.append(corr)
    
    conditional_pearson_r = float(np.mean(conditional_correlations)) if conditional_correlations else 0.0
    
    # 计算到达率（trips 数量 / 总步数）
    arrival_rate = n_trips / max_T if max_T > 0 else 0
    
    metrics = Phase3Metrics(
        n_agents=n_agents,
        n_arrived=n_trips,
        arrival_rate=arrival_rate,
        mean_steps=float(max_T * n_agents / n_trips) if n_trips > 0 else max_T,  # 平均每个 trip 的步数
        mean_speed=mean_speed,
        std_speed=std_speed,
        od_pearson_r=float(pearson_r),
        od_spearman_r=float(spearman_r),
    )
    
    # 可视化
    if save_figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. OD 分布散点图
        ax = axes[0, 0]
        ax.scatter(real_dist, sim_dist, alpha=0.5, s=10)
        max_val = max(real_dist.max(), sim_dist.max()) if len(real_dist) > 0 else 1
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1)
        ax.set_xlabel("Real OD Distribution")
        ax.set_ylabel("Simulated OD Distribution")
        ax.set_title(f"OD Distribution Comparison\nPearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}")
        
        # 2. 速度分布
        ax = axes[0, 1]
        if len(valid_speed) > 0:
            sample_speed = valid_speed.flatten()
            if len(sample_speed) > 100000:
                sample_speed = np.random.choice(sample_speed, 100000, replace=False)
            ax.hist(sample_speed, bins=50, density=True, alpha=0.7, edgecolor="black")
            ax.axvline(mean_speed, color="r", linestyle="--", label=f"Mean: {mean_speed:.2f}")
            ax.legend()
        ax.set_xlabel("Speed (pixels/step)")
        ax.set_ylabel("Density")
        ax.set_title("Speed Distribution")
        
        # 3. 目的地分布
        ax = axes[1, 0]
        dest_counts = {}
        for _, d in trips:
            dest_counts[d] = dest_counts.get(d, 0) + 1
        sink_ids = sorted(dest_counts.keys())
        counts = [dest_counts[s] for s in sink_ids]
        ax.bar(sink_ids, counts, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Sink ID")
        ax.set_ylabel("Trip Count")
        ax.set_title(f"Destination Distribution ({n_trips:,} trips)")
        
        # 4. 样本轨迹
        ax = axes[1, 1]
        walkable_mask = np.load(paths.walkable_mask)
        ax.imshow(walkable_mask, cmap="Greys", alpha=0.5, origin="upper")
        
        # 绘制几条随机轨迹（转置后 positions 是 [T, N, 2]）
        sample_indices = np.random.choice(n_agents, min(20, n_agents), replace=False)
        for idx in sample_indices:
            # positions[:, idx, :] 是 agent idx 的轨迹
            traj = positions[:1000, idx, :]  # 只取前1000步
            ax.plot(traj[:, 0], traj[:, 1], linewidth=0.5, alpha=0.7)
        ax.set_title("Sample Trajectories (first 1000 steps)")
        ax.axis("off")
        
        plt.tight_layout()
        fig.savefig(paths.figures_dir / "phase3_trajectories.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {paths.figures_dir / 'phase3_trajectories.png'}")
    
    # 保存指标
    np.savez(
        paths.metrics_dir / "phase3_metrics.npz",
        n_agents=metrics.n_agents,
        n_arrived=metrics.n_arrived,
        arrival_rate=metrics.arrival_rate,
        mean_steps=metrics.mean_steps,
        mean_speed=metrics.mean_speed,
        std_speed=metrics.std_speed,
        od_pearson_r=metrics.od_pearson_r,
        od_spearman_r=metrics.od_spearman_r,
        real_dist=real_dist,
        sim_dist=sim_dist,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Phase 3 Validation Results")
    print("=" * 60)
    print(f"  Total agents: {n_agents:,}")
    print(f"  Total time steps: {max_T:,}")
    print(f"  Total trips: {n_trips:,}")
    print(f"  Trips per agent: {n_trips / n_agents:.1f}")
    print(f"  Avg steps per trip: {max_T * n_agents / n_trips:.0f}" if n_trips > 0 else "  Avg steps per trip: N/A")
    print(f"  Mean speed: {mean_speed:.2f} ± {std_speed:.2f} px/step")
    print(f"  OD Distribution:")
    print(f"    Overall Pearson r = {pearson_r:.4f}")
    print(f"    Overall Spearman r = {spearman_r:.4f}")
    print(f"    Conditional Pearson r = {conditional_pearson_r:.4f} (per-origin avg)")
    
    if conditional_pearson_r > 0.9:
        print("  ✅ Conditional OD distribution correlation excellent")
    elif conditional_pearson_r > 0.7:
        print("  ✅ Conditional OD distribution correlation good")
    elif conditional_pearson_r > 0.5:
        print("  ⚠️ Conditional OD distribution correlation moderate")
    else:
        print("  ❌ Conditional OD distribution correlation poor")
    
    return metrics


if __name__ == "__main__":
    validate_phase3()
