"""Phase 3 正确评估"""
import numpy as np
import h5py
from pathlib import Path

# 加载有效样本
valid_trips = np.load('data/output/valid_eval_trips.npy', allow_pickle=True)

# 加载数据
with h5py.File('data/output/trajectories.h5', 'r') as f:
    pos = f['positions'][:]

nav_fields_dir = Path('data/processed/nav_fields')
dist_fields = {}
for i in range(35):
    path = nav_fields_dir / f'nav_field_{i:03d}.npz'
    if path.exists():
        dist_fields[i] = np.load(path)['distance_field']

H, W = dist_fields[0].shape

print('=== Phase 3 正确评估 (100 个样本) ===')
np.random.seed(42)
sample_indices = np.random.choice(len(valid_trips), 100, replace=False)

results = []
for trip_idx in sample_indices:
    trip = valid_trips[trip_idx]
    agent = trip['agent']
    t_start = trip['t_start']
    target = trip['target']
    start_dist = trip['start_dist']
    
    # 300 步终点
    t_end = min(trip['t_end'], t_start + 300)
    yf, xf = pos[t_end-1, agent]
    yf, xf = int(np.clip(yf, 0, H-1)), int(np.clip(xf, 0, W-1))
    end_dist = dist_fields[target][yf, xf]
    
    # 轨迹
    traj = pos[t_start:t_end, agent, :]
    vel = np.diff(traj, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    
    # 方向变化
    valid = speed > 0.1
    if valid.sum() >= 2:
        vel_valid = vel[valid]
        angles = np.arctan2(vel_valid[:, 1], vel_valid[:, 0])
        ad = np.abs(np.diff(angles))
        ad = np.minimum(ad, 2*np.pi - ad)
        smoothness = np.degrees(np.mean(ad))
    else:
        smoothness = 0
    
    results.append({
        'start_dist': start_dist,
        'end_dist': end_dist,
        'change': end_dist - start_dist,
        'speed': np.mean(speed[speed > 0.01]) if (speed > 0.01).any() else 0,
        'smoothness': smoothness
    })

# 汇总
changes = [r['change'] for r in results]
end_dists = [r['end_dist'] for r in results]
speeds = [r['speed'] for r in results]
smoothness = [r['smoothness'] for r in results]
start_dists = [r['start_dist'] for r in results]

print(f'样本数: {len(results)}')
print(f'起点距离: {np.mean(start_dists):.1f} px')
print(f'终点距离: {np.mean(end_dists):.1f} +/- {np.std(end_dists):.1f} px')
print(f'距离变化: {np.mean(changes):.1f} +/- {np.std(changes):.1f} px')
print(f'靠近率: {np.mean(np.array(changes) < 0)*100:.1f}%')
print(f'到达率: {np.mean(np.array(end_dists) < 10)*100:.1f}%')
print(f'速度: {np.mean(speeds):.2f} +/- {np.std(speeds):.2f} px/step')
print(f'平滑度: {np.mean(smoothness):.1f} deg/step')
