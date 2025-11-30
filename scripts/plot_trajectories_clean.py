"""
生成去除瞬移线的轨迹图：
- 仅绘制前若干粒子，按阈值断开大跳跃（重生导致的瞬移）
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 确保项目根目录在 sys.path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.phase3 import config


def plot_clean(n_agents=500, jump_thr=10.0, frame_stride=1, out_path=None):
    path = Path(config.TRAJ_PATH)
    if not path.exists():
        print("file not found:", path)
        return
    with h5py.File(path, "r") as f:
        pos = f["positions"]
        T, N, _ = pos.shape
        n_use = min(n_agents, N)
        traj = pos[:, :n_use, :]  # (T, n_use, 2)

    mask = np.load(config.MASK_PATH)
    frames = np.arange(0, T, frame_stride)

    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap="gray", alpha=0.3, origin="lower")

    for i in range(n_use):
        xy = traj[frames, i]
        diffs = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
        jump_idx = np.where(diffs > jump_thr)[0]
        start = 0
        for j in jump_idx:
            end = j + 1
            if end - start > 1:
                plt.plot(xy[start:end, 1], xy[start:end, 0], linewidth=0.5, alpha=0.6)
            start = end + 1
        if len(xy) - start > 1:
            plt.plot(xy[start:, 1], xy[start:, 0], linewidth=0.5, alpha=0.6)

    plt.gca().invert_yaxis()
    plt.title(f"Trajectories (n={n_use}, jump>{jump_thr} px broken)")
    plt.xlabel("x (pix)")
    plt.ylabel("y (pix)")
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = out_path or (config.OUTPUT_DIR / "trajectories_clean.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_clean()
