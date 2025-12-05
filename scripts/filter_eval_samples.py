"""筛选合理的评估样本"""
import numpy as np
import h5py
from pathlib import Path

with h5py.File('data/output/trajectories.h5', 'r') as f:
    pos = f['positions'][:]
    dest = f['destinations'][:]

nav_fields_dir = Path('data/processed/nav_fields')
sink_dist = {}
for i in range(35):
    path = nav_fields_dir / f'nav_field_{i:03d}.npz'
    if path.exists():
        sink_dist[i] = np.load(path)['distance_field']

H, W = sink_dist[0].shape

print('=== 筛选合理评估样本 ===')
print('条件: 起点距离 > 50, 且 trip 长度 >= 100')

valid_trips = []
for agent in range(min(1000, pos.shape[1])):
    dest_seq = dest[:, agent]
    changes = np.where(np.diff(dest_seq) != 0)[0] + 1
    
    segment_starts = np.concatenate([[0], changes])
    segment_ends = np.concatenate([changes, [pos.shape[0]]])
    
    for t_start, t_end in zip(segment_starts, segment_ends):
        trip_len = t_end - t_start
        if trip_len < 100:
            continue
        
        target = dest[t_start, agent]
        y0, x0 = pos[t_start, agent]
        y0, x0 = int(np.clip(y0, 0, H-1)), int(np.clip(x0, 0, W-1))
        
        start_dist = sink_dist[target][y0, x0]
        if start_dist < 50:
            continue
        
        # 计算终点距离 (最多 300 步)
        t_eval_end = min(t_end-1, t_start+299)
        yf, xf = pos[t_eval_end, agent]
        yf, xf = int(np.clip(yf, 0, H-1)), int(np.clip(xf, 0, W-1))
        end_dist = sink_dist[target][yf, xf]
        
        valid_trips.append({
            'agent': agent,
            't_start': t_start,
            't_end': t_end,
            'target': target,
            'start_dist': start_dist,
            'end_dist': end_dist,
            'trip_len': trip_len
        })

print(f'找到 {len(valid_trips)} 个有效 trips')

# 统计
start_dists = [t['start_dist'] for t in valid_trips]
end_dists = [t['end_dist'] for t in valid_trips]
changes = [t['end_dist'] - t['start_dist'] for t in valid_trips]

print(f'\n起点距离: mean={np.mean(start_dists):.1f}, std={np.std(start_dists):.1f}')
print(f'终点距离: mean={np.mean(end_dists):.1f}, std={np.std(end_dists):.1f}')
print(f'距离变化: mean={np.mean(changes):.1f}, std={np.std(changes):.1f}')
print(f'靠近率 (change < 0): {np.mean(np.array(changes) < 0)*100:.1f}%')
print(f'到达率 (end < 10): {np.mean(np.array(end_dists) < 10)*100:.1f}%')

# 示例
print('\n示例 trips:')
for i in [0, 1, 2, 10, 50]:
    if i < len(valid_trips):
        t = valid_trips[i]
        print(f"  Agent {t['agent']}: target={t['target']}, dist {t['start_dist']:.0f} -> {t['end_dist']:.0f}, len={t['trip_len']}")

# 保存有效样本供后续使用
print(f'\n保存有效 trips 到 data/output/valid_eval_trips.npy')
np.save('data/output/valid_eval_trips.npy', valid_trips)
