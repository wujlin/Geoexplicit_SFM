"""
多进程预计算有效样本索引
过滤掉低速样本，保存到 npy 文件供训练使用
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def filter_agent(args):
    """处理单个 agent 的过滤"""
    agent, agent_vel, history, future, stride, windows_per_agent, min_speed = args
    
    valid_indices = []
    for w in range(windows_per_agent):
        t_idx = w * stride
        # 获取 future 时间段的速度
        future_vel = agent_vel[t_idx + history : t_idx + history + future, :]
        avg_speed = np.linalg.norm(future_vel, axis=-1).mean()
        
        if avg_speed >= min_speed:
            valid_indices.append((int(agent), t_idx))
    
    return valid_indices


def main():
    parser = argparse.ArgumentParser(description="Precompute valid sample indices")
    parser.add_argument("--h5_path", type=str, default="data/output/trajectories.h5")
    parser.add_argument("--output", type=str, default="data/output/valid_indices.npy")
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--future", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min_speed", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (default: CPU count)")
    
    args = parser.parse_args()
    
    h5_path = Path(args.h5_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {h5_path}...")
    
    with h5py.File(h5_path, "r") as f:
        T, N, _ = f["positions"].shape
        print(f"Data shape: T={T}, N={N}")
        
        # 加载速度数据
        print("Loading velocities into memory...")
        if "velocities" in f:
            velocities = f["velocities"][:]
        else:
            positions = f["positions"][:]
            velocities = np.diff(positions, axis=0)
            velocities = np.concatenate([velocities, velocities[-1:]], axis=0)
    
    print(f"Velocities shape: {velocities.shape}")
    
    # 计算窗口数
    windows_per_agent = (T - (args.history + args.future)) // args.stride
    total_possible = N * windows_per_agent
    print(f"Total possible samples: {total_possible:,}")
    print(f"Min speed threshold: {args.min_speed}")
    
    # 准备多进程参数
    num_workers = args.workers or mp.cpu_count()
    print(f"Using {num_workers} workers...")
    
    # 准备每个 agent 的参数
    task_args = [
        (agent, velocities[:, agent, :].copy(), args.history, args.future, 
         args.stride, windows_per_agent, args.min_speed)
        for agent in range(N)
    ]
    
    # 多进程处理
    valid_indices = []
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(filter_agent, task_args),
            total=N,
            desc="Filtering agents"
        ))
    
    # 合并结果
    for agent_result in results:
        valid_indices.extend(agent_result)
    
    # 转换为 numpy 数组并保存
    valid_indices = np.array(valid_indices, dtype=np.int64)
    
    keep_ratio = len(valid_indices) / total_possible * 100
    print(f"\nValid samples: {len(valid_indices):,} / {total_possible:,} ({keep_ratio:.1f}%)")
    
    np.save(output_path, valid_indices)
    print(f"Saved to {output_path}")
    
    # 打印速度分布统计
    print("\n=== Speed distribution of valid samples ===")
    speeds = []
    for agent, t_idx in valid_indices[:10000]:  # 采样检查
        future_vel = velocities[t_idx + args.history : t_idx + args.history + args.future, agent, :]
        avg_speed = np.linalg.norm(future_vel, axis=-1).mean()
        speeds.append(avg_speed)
    speeds = np.array(speeds)
    print(f"Mean speed: {speeds.mean():.4f}")
    print(f"Median speed: {np.median(speeds):.4f}")
    print(f"Min speed: {speeds.min():.4f}")
    print(f"Max speed: {speeds.max():.4f}")


if __name__ == "__main__":
    main()
