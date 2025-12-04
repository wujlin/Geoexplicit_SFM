"""检查个体导航场和全局导航场的差异"""
import numpy as np
from pathlib import Path
import json

nav_fields_dir = Path('data/processed/nav_fields')

with open(nav_fields_dir / 'nav_fields_index.json') as f:
    index = json.load(f)

print('个体导航场检查')
print('=' * 60)
sink_ids = index['sink_ids']
print(f'sink_ids: {sink_ids[:10]}...')
print(f'num_sinks: {index["num_sinks"]}')

# 加载全局导航场
nav_global = np.load('data/processed/nav_baseline.npz')
nav_g = np.stack([nav_global['nav_y'], nav_global['nav_x']], axis=0)

# 加载 dest=33 的导航场
sink_id = 33
data = np.load(nav_fields_dir / f'nav_field_{sink_id:03d}.npz')
nav_33 = np.stack([data['nav_y'], data['nav_x']], axis=0)
print(f'\nSink {sink_id} 导航场:')
print(f'  shape: {nav_33.shape}')

# 检查位置 (823, 1021) 的值
y, x = 823, 1021
print(f'  nav_33[:, {y}, {x}]: {nav_33[:, y, x]}')
print(f'  nav_global[:, {y}, {x}]: {nav_g[:, y, x]}')

# 再检查 dest=13 的导航场
sink_id = 13
data = np.load(nav_fields_dir / f'nav_field_{sink_id:03d}.npz')
nav_13 = np.stack([data['nav_y'], data['nav_x']], axis=0)
print(f'\nSink {sink_id} 导航场:')
y, x = 740, 495
print(f'  nav_13[:, {y}, {x}]: {nav_13[:, y, x]}')
print(f'  nav_global[:, {y}, {x}]: {nav_g[:, y, x]}')

y, x = 737, 678
print(f'  nav_13[:, {y}, {x}]: {nav_13[:, y, x]}')
print(f'  nav_global[:, {y}, {x}]: {nav_g[:, y, x]}')

# 统计不同个体导航场之间的差异
print('\n' + '=' * 60)
print('不同 sink 导航场差异统计')
nav_fields = {}
for sid in sink_ids[:5]:  # 只检查前 5 个
    data = np.load(nav_fields_dir / f'nav_field_{sid:03d}.npz')
    nav_fields[sid] = np.stack([data['nav_y'], data['nav_x']], axis=0)

# 在一些位置比较
test_positions = [(500, 700), (600, 800), (700, 900), (400, 600)]
for y, x in test_positions:
    print(f'\n位置 ({y}, {x}):')
    print(f'  global: [{nav_g[0,y,x]:.3f}, {nav_g[1,y,x]:.3f}]')
    for sid, nav in nav_fields.items():
        print(f'  sink {sid}: [{nav[0,y,x]:.3f}, {nav[1,y,x]:.3f}]')
