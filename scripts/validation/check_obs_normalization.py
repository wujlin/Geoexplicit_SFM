"""检查 obs 归一化后的数值范围"""

import sys
import numpy as np
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
from pathlib import Path
import importlib.util
import h5py

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

phase4_root = Path('src/phase4')
normalizer_mod = _import_module('normalizer', phase4_root / 'data' / 'normalizer.py')

ckpt = torch.load('data/output/phase4_checkpoints/best.pt', map_location='cpu', weights_only=False)

obs_normalizer = normalizer_mod.ObsNormalizer()
obs_normalizer.load_state_dict(ckpt['obs_normalizer'])

print('ObsNormalizer 参数检查')
print('=' * 60)

print('Position normalizer:')
print(f'  mean: {obs_normalizer.pos_normalizer.mean}')
print(f'  std: {obs_normalizer.pos_normalizer.std}')

print()
print('Velocity normalizer:')
print(f'  mean: {obs_normalizer.vel_normalizer.mean}')
print(f'  std: {obs_normalizer.vel_normalizer.std}')

print()
print('Nav normalizer:')
nav_state = obs_normalizer.nav_normalizer.state_dict()
print(f'  state: {nav_state}')

# 测试一个真实样本的归一化
print()
print('=' * 60)
print('测试真实样本归一化')

# 加载数据
h5 = h5py.File('data/output/trajectories.h5', 'r')
pos = h5['positions'][:]
vel = h5['velocities'][:]
dest = h5['destinations'][:]

nav_fields_dir = Path('data/processed/nav_fields')
nav_files = sorted(nav_fields_dir.glob('nav_field_*.npz'))
nav_data = {}
for f in nav_files:
    d = np.load(f)
    sink_id = int(f.stem.split('_')[2])
    nav_data[sink_id] = np.stack([d['nav_y'], d['nav_x']], axis=0)

H, W = 1365, 1435

# 选择一个有效样本
t0, agent = 1000, 100
d = dest[t0, agent]

obs_raw = np.zeros((2, 6), dtype=np.float32)
for i in range(2):
    t = t0 + i
    p = pos[t, agent]
    v = vel[t, agent]
    y, x = int(np.clip(p[0], 0, H-1)), int(np.clip(p[1], 0, W-1))
    nav = nav_data[d][:, y, x] if d in nav_data else np.zeros(2)
    obs_raw[i] = np.concatenate([p, v, nav])

print(f'原始 obs (agent={agent}, t0={t0}, dest={d}):')
print(f'  obs[0]: pos=[{obs_raw[0,0]:.1f}, {obs_raw[0,1]:.1f}], vel=[{obs_raw[0,2]:.3f}, {obs_raw[0,3]:.3f}], nav=[{obs_raw[0,4]:.3f}, {obs_raw[0,5]:.3f}]')
print(f'  obs[1]: pos=[{obs_raw[1,0]:.1f}, {obs_raw[1,1]:.1f}], vel=[{obs_raw[1,2]:.3f}, {obs_raw[1,3]:.3f}], nav=[{obs_raw[1,4]:.3f}, {obs_raw[1,5]:.3f}]')

# 归一化
obs_tensor = torch.from_numpy(obs_raw).unsqueeze(0)
obs_normed = obs_normalizer.transform(obs_tensor)

print()
print('归一化后 obs:')
obs_np = obs_normed.numpy()[0]
print(f'  obs[0]: pos=[{obs_np[0,0]:.3f}, {obs_np[0,1]:.3f}], vel=[{obs_np[0,2]:.3f}, {obs_np[0,3]:.3f}], nav=[{obs_np[0,4]:.3f}, {obs_np[0,5]:.3f}]')
print(f'  obs[1]: pos=[{obs_np[1,0]:.3f}, {obs_np[1,1]:.3f}], vel=[{obs_np[1,2]:.3f}, {obs_np[1,3]:.3f}], nav=[{obs_np[1,4]:.3f}, {obs_np[1,5]:.3f}]')

print()
print('各维度归一化后的范围:')
print(f'  pos: [{obs_np[:,:2].min():.2f}, {obs_np[:,:2].max():.2f}]')
print(f'  vel: [{obs_np[:,2:4].min():.2f}, {obs_np[:,2:4].max():.2f}]')
print(f'  nav: [{obs_np[:,4:6].min():.2f}, {obs_np[:,4:6].max():.2f}]')

# 展平后的 global_cond
global_cond = obs_np.flatten()
print()
print('展平后的 global_cond (12维):')
print(f'  shape: {global_cond.shape}')
print(f'  values: {global_cond}')

h5.close()

# ============================================================
# 检查不同方向的 nav 归一化后的差异
# ============================================================
print()
print('=' * 60)
print('不同方向 nav 归一化后的差异')

# 模拟 4 个方向
directions = [
    ('上', np.array([-1, 0])),
    ('下', np.array([1, 0])),
    ('左', np.array([0, -1])),
    ('右', np.array([0, 1])),
]

for name, nav in directions:
    nav_tensor = torch.from_numpy(nav.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
    nav_normed = obs_normalizer.nav_normalizer.transform(nav_tensor)
    print(f'  {name}: {nav.tolist()} -> {nav_normed.numpy().flatten().tolist()}')
