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
print(f"min: {an['normalizer']['min_val']}")
print(f"max: {an['normalizer']['max_val']}")

print('\n=== Obs Normalizer ===')
on = ckpt['obs_normalizer']
print(f"mode: {on['mode']}")
print(f"pos min: {on['pos_normalizer']['min_val']}")
print(f"pos max: {on['pos_normalizer']['max_val']}")
print(f"vel min: {on['vel_normalizer']['min_val']}")
print(f"vel max: {on['vel_normalizer']['max_val']}")
if 'nav_normalizer' in on:
    print(f"nav min: {on['nav_normalizer']['min_val']}")
    print(f"nav max: {on['nav_normalizer']['max_val']}")

# 计算 action 的反归一化范围
action_min = an['normalizer']['min_val']
action_max = an['normalizer']['max_val']
print(f'\n=== 归一化后 [-1,1] 对应原始值 ===')
print(f'-1 对应: {action_min}')
print(f'+1 对应: {action_max}')
print(f'0 对应 (中心): {(action_min + action_max) / 2}')

# 模拟: 如果模型输出集中在 0 附近，反归一化后会是什么值？
print(f'\n=== 如果模型预测 ~0 (中心) ===')
center = (action_min + action_max) / 2
print(f'预测速度: {center}')
print(f'预测速度幅度: {np.linalg.norm(center):.4f}')

# 检查 GT 速度分布
print(f'\n=== GT 速度分布 ===')
import h5py
with h5py.File(PROJECT_ROOT / 'data/output/trajectories.h5', 'r') as f:
    vel = f['velocities'][:]
    
speeds = np.sqrt((vel**2).sum(axis=2))
print(f'速度 mean: {speeds.mean():.4f}')
print(f'速度 std: {speeds.std():.4f}')
print(f'速度 min: {speeds.min():.4f}')
print(f'速度 max: {speeds.max():.4f}')

# 如果模型总是预测 0（归一化空间），反归一化后速度是多少？
if isinstance(action_min, np.ndarray):
    pred_zero = (action_min + action_max) / 2
    pred_speed = np.linalg.norm(pred_zero)
    print(f'\n如果模型预测归一化空间的 0，反归一化后速度: {pred_speed:.4f}')
