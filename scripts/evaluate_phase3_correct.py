"""
Phase 3 vs Phase 4 正确评估脚本

关键修正：使用到目标 sink 的距离，而非到最近 sink 的距离
"""
import numpy as np
import h5py
from pathlib import Path

# 加载数据
print("=== 加载数据 ===")
mask = np.load('data/processed/walkable_mask.npy')
H, W = mask.shape
print(f"地图尺寸: {H} x {W}")

# 加载每个 sink 的距离场
print("\n加载每个 sink 的距离场...")
nav_fields_dir = Path('data/processed/nav_fields')
sink_dist_fields = {}
for i in range(35):
    path = nav_fields_dir / f'nav_field_{i:03d}.npz'
    if path.exists():
        data = np.load(path)
        sink_dist_fields[i] = data['distance_field']
print(f"加载了 {len(sink_dist_fields)} 个 sink 的距离场")

# 加载 Phase 3 轨迹
with h5py.File('data/output/trajectories.h5', 'r') as f:
    pos = f['positions'][:]  # (T, N, 2)
    dest = f['destinations'][:]  # (T, N)
print(f"Phase 3 轨迹: {pos.shape}")

# 辅助函数
def get_dist_to_sink(y, x, sink_id):
    """获取位置到指定 sink 的距离"""
    if sink_id not in sink_dist_fields:
        return np.inf
    y = int(np.clip(y, 0, H-1))
    x = int(np.clip(x, 0, W-1))
    return sink_dist_fields[sink_id][y, x]

# 分析 Phase 3 的 trips
print("\n=== Phase 3 正确评估 ===")

# 找所有有效的 trips
all_trips = []
for agent_id in range(min(500, pos.shape[1])):  # 采样 500 个 agent
    dest_seq = dest[:, agent_id]
    changes = np.where(np.diff(dest_seq) != 0)[0] + 1
    
    # 每个 trip 的边界
    segment_starts = np.concatenate([[0], changes])
    segment_ends = np.concatenate([changes, [pos.shape[0]]])
    
    for t_start, t_end in zip(segment_starts, segment_ends):
        if t_end - t_start < 50:  # 跳过太短的 trip
            continue
        
        target_sink = dest[t_start, agent_id]
        
        # 起点和终点到目标 sink 的距离
        y0, x0 = pos[t_start, agent_id]
        yf, xf = pos[t_end-1, agent_id]
        
        dist_start = get_dist_to_sink(y0, x0, target_sink)
        dist_end = get_dist_to_sink(yf, xf, target_sink)
        
        if np.isinf(dist_start) or np.isinf(dist_end):
            continue
        
        all_trips.append({
            'agent': agent_id,
            't_start': t_start,
            't_end': t_end,
            'target_sink': target_sink,
            'dist_start': dist_start,
            'dist_end': dist_end,
            'dist_change': dist_end - dist_start,
        })

print(f"找到 {len(all_trips)} 个有效 trips")

# 统计
dist_changes = [t['dist_change'] for t in all_trips]
dist_starts = [t['dist_start'] for t in all_trips]
dist_ends = [t['dist_end'] for t in all_trips]

print(f"\n起点到目标 sink 距离: mean={np.mean(dist_starts):.1f}, std={np.std(dist_starts):.1f}")
print(f"终点到目标 sink 距离: mean={np.mean(dist_ends):.1f}, std={np.std(dist_ends):.1f}")
print(f"距离变化 (负=靠近): mean={np.mean(dist_changes):.1f}, std={np.std(dist_changes):.1f}")
print(f"靠近目标率 (change < 0): {np.mean(np.array(dist_changes) < 0)*100:.1f}%")
print(f"到达率 (dist_end < 10): {np.mean(np.array(dist_ends) < 10)*100:.1f}%")

# 分析为什么起点距离不为 0
print("\n=== 起点距离分析 ===")
print("注意: Phase 3 从一个 sink 出发，去往另一个 sink")
print("起点到目标 sink 的距离 = 两个 sink 之间的距离")

# 随机看几个 trip
print("\n示例 trips:")
for i in [0, 1, 2, 10, 50]:
    if i < len(all_trips):
        t = all_trips[i]
        print(f"  Trip {i}: agent={t['agent']}, target={t['target_sink']}, "
              f"dist {t['dist_start']:.0f} -> {t['dist_end']:.0f} = {t['dist_change']:.0f}")
