"""分析 Phase 1 sinks 和 Phase 2 target_density"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 sinks
sinks = pd.read_csv('data/processed/sinks_phase1.csv')
sinks_sorted = sinks.sort_values('total_flow', ascending=False)
total_flow = sinks['total_flow'].sum()

print("=" * 50)
print("Phase 1 Sinks Analysis")
print("=" * 50)
print(f"Total sinks: {len(sinks)}")
print(f"Total flow: {total_flow:,}")
print()

print("Top 10 sinks by flow:")
cumsum = 0
for idx, (_, row) in enumerate(sinks_sorted.head(10).iterrows()):
    pct = row['total_flow'] / total_flow * 100
    cumsum += row['total_flow']
    cum_pct = cumsum / total_flow * 100
    print(f"  {idx+1}. Cluster {int(row['cluster_id']):2d}: "
          f"flow={int(row['total_flow']):>7,} ({pct:4.1f}%) "
          f"cum={cum_pct:5.1f}% "
          f"pts={int(row['n_points']):2d} "
          f"loc=({row['lat']:.3f}, {row['lon']:.3f})")

print()
print(f"Top 10 cover: {sinks_sorted.head(10)['total_flow'].sum()/total_flow*100:.1f}%")
print(f"Max/Min flow ratio: {sinks['total_flow'].max()/sinks['total_flow'].min():.0f}x")

# 空间范围
print()
print("Spatial extent:")
print(f"  Lat: [{sinks['lat'].min():.4f}, {sinks['lat'].max():.4f}]")
print(f"  Lon: [{sinks['lon'].min():.4f}, {sinks['lon'].max():.4f}]")

# 检查 target_density
print()
print("=" * 50)
print("Phase 2 Target Density Analysis")
print("=" * 50)

target = np.load('data/processed/target_density.npy')
mask = np.load('data/processed/walkable_mask.npy')

print(f"Shape: {target.shape}")
print(f"Value range: [{target.min():.6f}, {target.max():.6f}]")
print(f"Non-zero pixels: {(target > 0).sum()} ({(target > 0).sum()/target.size*100:.2f}%)")

# Sink 区域 (高密度)
threshold = 0.01
sink_pixels = target > threshold
print(f"High density (>{threshold}): {sink_pixels.sum()} pixels ({sink_pixels.sum()/target.size*100:.3f}%)")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Sinks 空间分布
ax = axes[0]
ax.scatter(sinks['lon'], sinks['lat'], s=sinks['total_flow']/1000, alpha=0.6, c='red')
for _, row in sinks_sorted.head(5).iterrows():
    ax.annotate(f"C{int(row['cluster_id'])}", (row['lon'], row['lat']), fontsize=8)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Phase 1 Sinks (size=flow)')
ax.set_aspect('equal')

# 2. Target density
ax = axes[1]
im = ax.imshow(target, cmap='hot', origin='upper')
plt.colorbar(im, ax=ax, label='density')
ax.set_title('Target Density')

# 3. Walkable mask with sink overlay
ax = axes[2]
ax.imshow(mask, cmap='gray', origin='upper')
# 找到高密度区域的位置
sink_y, sink_x = np.where(target > threshold)
if len(sink_y) > 0:
    ax.scatter(sink_x, sink_y, c='red', s=1, alpha=0.5, label=f'sink pixels ({len(sink_y)})')
ax.set_title('Walkable + Sink regions')
ax.legend()

plt.tight_layout()
plt.savefig('data/output/sinks_analysis.png', dpi=150)
print(f"\nSaved: data/output/sinks_analysis.png")
