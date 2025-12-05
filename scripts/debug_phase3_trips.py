"""分析 Phase 3 的单个 trip"""
import numpy as np
import h5py

dist_field = np.load('data/processed/distance_field.npy')
with h5py.File('data/output/trajectories.h5', 'r') as f:
    pos = f['positions'][:]
    dest = f['destinations'][:]

H, W = dist_field.shape

# 现在理解了问题：Phase 3 仿真从 sink 出发，走向另一个 sink
# 所以 t=0 时距离=0 是对的！
# respawn 后也是到达 sink（距离接近 0）再瞬移到下一个 sink（距离也是 0）

# 让我们找一个完整的 trip（从 respawn 到下一个 respawn）
print('=== 分析一个完整的 trip ===')
agent = 1
dest_seq = dest[:, agent]
changes = np.where(np.diff(dest_seq) != 0)[0] + 1

# 第一个 trip: t=0 到第一次 respawn
t_start = 0
t_end = changes[0] if len(changes) > 0 else pos.shape[0]
print(f'Trip 1: t={t_start} to t={t_end} (length={t_end - t_start})')

# 这个 trip 中每个时刻的距离
dists = []
for t in range(t_start, min(t_end, t_start + 100)):
    y = int(np.clip(pos[t, agent, 0], 0, H-1))
    x = int(np.clip(pos[t, agent, 1], 0, W-1))
    dists.append(dist_field[y, x])
print(f'距离序列 (前100步): {dists[:20]}...{dists[-10:] if len(dists) > 20 else ""}')
print(f'起点距离: {dists[0]:.1f}, 终点距离: {dists[-1]:.1f}')
print(f'距离变化: {dists[-1] - dists[0]:.1f} (正=远离, 负=靠近)')

# 关键发现：起点距离=0（在sink），终点距离≈0（到达另一个sink）
# 但中间过程会先远离再靠近！

# 统计整个 trip 的距离变化
print(f'\n整个 trip 的距离统计:')
full_dists = []
for t in range(t_start, t_end):
    y = int(np.clip(pos[t, agent, 0], 0, H-1))
    x = int(np.clip(pos[t, agent, 1], 0, W-1))
    full_dists.append(dist_field[y, x])
full_dists = np.array(full_dists)
print(f'最大距离: {full_dists.max():.1f} (在 t={t_start + np.argmax(full_dists)})')
print(f'平均距离: {full_dists.mean():.1f}')

# 第二个 trip
if len(changes) > 1:
    t_start = changes[0]
    t_end = changes[1]
    print(f'\n=== Trip 2: t={t_start} to t={t_end} (length={t_end - t_start}) ===')
    
    full_dists = []
    for t in range(t_start, t_end):
        y = int(np.clip(pos[t, agent, 0], 0, H-1))
        x = int(np.clip(pos[t, agent, 1], 0, W-1))
        full_dists.append(dist_field[y, x])
    full_dists = np.array(full_dists)
    print(f'起点距离: {full_dists[0]:.1f}, 终点距离: {full_dists[-1]:.1f}')
    print(f'最大距离: {full_dists.max():.1f}')
    print(f'距离变化: {full_dists[-1] - full_dists[0]:.1f}')

# 关键分析：理解 Phase 3 的 "distance_field" 含义
print('\n=== 关键发现 ===')
print('Phase 3 使用的 distance_field 是到最近 sink 的距离，不是到目标 sink 的距离！')
print('所以:')
print('  - 起点在 sink A，distance = 0')
print('  - 移动过程中，distance 会先增加（离开 A），然后减小（靠近 B）')
print('  - 但如果 A 和 B 距离很近，distance 可能一直很小')
print('')
print('评估时应该用到目标 sink 的距离，而非到最近 sink 的距离！')
