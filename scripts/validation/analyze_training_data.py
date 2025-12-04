"""分析训练数据的分布"""

import sys
import numpy as np
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import h5py
from pathlib import Path

h5_path = Path('data/output/trajectories.h5')
h5 = h5py.File(h5_path, 'r')

vel = h5['velocities'][:]  # (T, N, 2)
pos = h5['positions'][:]
dest = h5['destinations'][:]

print("轨迹数据分析")
print("=" * 60)
print()

# 计算速度分布
vel_flat = vel.reshape(-1, 2)
speed = np.linalg.norm(vel_flat, axis=1)

print("速度分布:")
print(f"  总样本数: {len(speed):,}")
print(f"  速度 mean: {speed.mean():.4f}")
print(f"  速度 std: {speed.std():.4f}")
print(f"  速度 min: {speed.min():.4f}")
print(f"  速度 max: {speed.max():.4f}")
print()

# 按速度区间统计
bins = [0, 0.1, 0.5, 1.0, 1.5, 2.0, np.inf]
for i in range(len(bins) - 1):
    low, high = bins[i], bins[i + 1]
    count = np.sum((speed >= low) & (speed < high))
    ratio = count / len(speed) * 100
    print(f"  速度 [{low:.1f}, {high:.1f}): {count:,} ({ratio:.1f}%)")

print()
print("速度分量分布:")
print(f"  vel_y mean={vel_flat[:, 0].mean():.4f}, std={vel_flat[:, 0].std():.4f}")
print(f"  vel_x mean={vel_flat[:, 1].mean():.4f}, std={vel_flat[:, 1].std():.4f}")

print()
print("分析静止样本（速度 < 0.1）:")
static_mask = speed < 0.1
static_count = np.sum(static_mask)
print(f"  静止样本数: {static_count:,} ({static_count / len(speed) * 100:.1f}%)")

# 分析 action 的方向分布
moving_mask = speed > 0.1
vel_moving = vel_flat[moving_mask]
angles = np.arctan2(vel_moving[:, 0], vel_moving[:, 1])  # 使用 (y, x) 顺序

print()
print("运动样本方向分布 (度):")
angles_deg = np.degrees(angles)
hist, bin_edges = np.histogram(angles_deg, bins=8, range=(-180, 180))
for i in range(len(hist)):
    low, high = bin_edges[i], bin_edges[i + 1]
    print(f"  [{low:.0f}°, {high:.0f}°): {hist[i]:,} ({hist[i] / len(angles_deg) * 100:.1f}%)")

h5.close()

print()
print("=" * 60)
print("结论:")
if static_count / len(speed) > 0.3:
    print("  WARNING: 训练数据中有大量静止样本，可能导致模型学习'平均静止'")
else:
    print("  OK: 静止样本比例正常")