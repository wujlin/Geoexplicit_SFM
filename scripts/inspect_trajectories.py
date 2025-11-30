"""
快速自检 Phase3 生成的 trajectories.h5：
- 打印基本信息与统计
- 抽样若干轨迹绘制流向，查看噪声与方向是否合理
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def basic_stats(pos_ds, vel_ds=None, sample_frames=10):
    T, N, _ = pos_ds.shape
    print("positions shape:", pos_ds.shape)
    first = pos_ds[0]
    last = pos_ds[-1]
    print("first frame min/max:", float(first.min()), float(first.max()))
    
    # 计算总位移
    disp = last - first  # (N, 2)
    total_disp = np.sqrt((disp**2).sum(axis=1))
    print(f"total displacement: mean={total_disp.mean():.2f}, max={total_disp.max():.2f} pixels")
    print(f"avg speed over {T} frames: {total_disp.mean() / T:.4f} pix/frame")
    
    if vel_ds is not None:
        v = vel_ds[:sample_frames]
        speed = np.sqrt((v ** 2).sum(-1))
        print(
            f"speed mean (first {sample_frames} frames): {float(speed.mean()):.4f}, "
            f"max: {float(speed.max()):.4f}"
        )


def sample_and_plot(pos_ds, save_path="data/output/trajectories_sample.png", n_agents=20, frame_step=None):
    T, N, _ = pos_ds.shape
    
    # 自动选择采样间隔：确保每条轨迹至少有 50 个点
    if frame_step is None:
        frame_step = max(1, T // 100)
    
    agents_idx = np.random.choice(N, size=min(n_agents, N), replace=False)
    agents_idx.sort()  # h5py 要求索引递增
    frames = np.arange(0, T, frame_step)
    print(f"Sampling {len(agents_idx)} agents, {len(frames)} frames (step={frame_step})")
    
    # 取子集 (frames, agents_idx, 2)
    traj = np.zeros((len(frames), len(agents_idx), 2), dtype=np.float32)
    for fi, fidx in enumerate(frames):
        traj[fi] = pos_ds[fidx, agents_idx, :]
    
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, n_agents))
    
    for j in range(traj.shape[1]):
        xy = traj[:, j]
        # 检测大跳跃（重生）并分段绘制
        jumps = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
        jump_thr = 100  # 像素
        
        # 找到分段点
        break_points = np.where(jumps > jump_thr)[0] + 1
        segments = np.split(np.arange(len(xy)), break_points)
        
        for seg in segments:
            if len(seg) > 1:
                plt.plot(xy[seg, 1], xy[seg, 0], alpha=0.7, linewidth=1.5, color=colors[j])
        
        # 起点终点
        plt.scatter(xy[0, 1], xy[0, 0], s=30, c="green", marker="o", zorder=5)
        plt.scatter(xy[-1, 1], xy[-1, 0], s=30, c="red", marker="x", zorder=5)
    
    plt.gca().invert_yaxis()
    plt.title(f"Sampled trajectories (n={len(agents_idx)}, T={T}, step={frame_step})")
    plt.xlabel("x (pix)")
    plt.ylabel("y (pix)")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved trajectory sample plot: {save_path}")


def main():
    path = Path("data/output/trajectories.h5")
    if not path.exists():
        print("File not found:", path)
        return
    print("exists:", path.exists(), "size(MB):", path.stat().st_size / 1e6)
    with h5py.File(path, "r") as f:
        pos = f["positions"]
        vel = f["velocities"] if "velocities" in f else None
        basic_stats(pos, vel, sample_frames=10)
        sample_and_plot(pos, save_path="data/output/trajectories_sample.png", n_agents=30)


if __name__ == "__main__":
    main()
