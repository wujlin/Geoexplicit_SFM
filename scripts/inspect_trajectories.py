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
    print("positions shape:", pos_ds.shape)
    first = pos_ds[0]
    print("first frame min/max:", float(first.min()), float(first.max()))
    if vel_ds is not None:
        v = vel_ds[:sample_frames]
        speed = np.sqrt((v ** 2).sum(-1))
        print(
            f"speed mean (first {sample_frames} frames): {float(speed.mean()):.4f}, "
            f"max: {float(speed.max()):.4f}"
        )


def sample_and_plot(pos_ds, save_path="data/output/trajectories_sample.png", n_agents=20, frame_step=50):
    T, N, _ = pos_ds.shape
    agents_idx = np.random.choice(N, size=min(n_agents, N), replace=False)
    agents_idx.sort()  # h5py 要求索引递增
    frames = np.arange(0, T, frame_step)
    # 取子集 (frames, agents_idx, 2)
    # h5py 只支持 1D 索引；分两步取
    traj = np.zeros((len(frames), len(agents_idx), 2), dtype=np.float32)
    for fi, fidx in enumerate(frames):
        traj[fi] = pos_ds[fidx, agents_idx, :]
    plt.figure(figsize=(8, 8))
    for j in range(traj.shape[1]):
        xy = traj[:, j]
        plt.plot(xy[:, 1], xy[:, 0], alpha=0.6)
        plt.scatter(xy[0, 1], xy[0, 0], s=10, c="green", alpha=0.8)
        plt.scatter(xy[-1, 1], xy[-1, 0], s=10, c="red", alpha=0.8)
        # 断开大跳跃（可能是重生导致的瞬移），超过阈值的段不连线
        jumps = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
        jump_thr = 200  # 像素
        if (jumps > jump_thr).any():
            # 重画：只画小于阈值的连续段
            plt.plot(
                xy[:, 1],
                xy[:, 0],
                alpha=0.2,
                linestyle="--",
                linewidth=0.5,
                label=None,
            )
    plt.gca().invert_yaxis()
    plt.title(f"Sampled trajectories (n={len(agents_idx)}, every {frame_step} frames)")
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
        sample_and_plot(pos, save_path="data/output/trajectories_sample.png", n_agents=20, frame_step=50)


if __name__ == "__main__":
    main()
