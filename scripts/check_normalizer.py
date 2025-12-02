"""检查 Phase 4 模型归一化器状态"""
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ckpt = torch.load(PROJECT_ROOT / 'data/output/phase4_checkpoints/best.pt', 
                  map_location='cpu', weights_only=False)

print('=== Action Normalizer ===')
an = ckpt['action_normalizer']
print(f"mode: {an['mode']}")
if an['mode'] == 'zscore':
    print(f"mean: {an['normalizer']['mean']}")
    print(f"std: {an['normalizer']['std']}")
else:
    print(f"min: {an['normalizer']['min_val']}")
    print(f"max: {an['normalizer']['max_val']}")

print('\n=== Obs Normalizer ===')
on = ckpt['obs_normalizer']
print(f"mode: {on['mode']}")
if on['mode'] == 'zscore':
    print(f"pos mean: {on['pos_normalizer']['mean']}")
    print(f"pos std: {on['pos_normalizer']['std']}")
    print(f"vel mean: {on['vel_normalizer']['mean']}")
    print(f"vel std: {on['vel_normalizer']['std']}")
else:
    print(f"pos min: {on['pos_normalizer']['min_val']}")
    print(f"pos max: {on['pos_normalizer']['max_val']}")

# 检查 GT 速度分布
print('\n=== GT 速度分布 ===')
import h5py
with h5py.File(PROJECT_ROOT / 'data/output/trajectories.h5', 'r') as f:
    vel = f['velocities'][:]
    
speeds = np.sqrt((vel**2).sum(axis=2))
print(f'速度 mean: {speeds.mean():.4f}')
print(f'速度 std: {speeds.std():.4f}')

# zscore 归一化后，预测 0 对应原始均值
if an['mode'] == 'zscore':
    action_mean = an['normalizer']['mean']
    action_std = an['normalizer']['std']
    print(f'\n=== ZScore 归一化分析 ===')
    print(f'如果模型预测归一化空间的 0，反归一化后:')
    print(f'  velocity = {action_mean}')
    print(f'  speed = {np.linalg.norm(action_mean):.4f}')
