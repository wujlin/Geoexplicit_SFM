"""检查模型 condition 注入是否正常工作"""

import sys
import numpy as np
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
from pathlib import Path
import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

phase4_root = Path('src/phase4')
unet_mod = _import_module('unet1d', phase4_root / 'model' / 'unet1d.py')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('data/output/phase4_checkpoints/best.pt', map_location=device, weights_only=False)
config = ckpt['config']

# 创建一个新的随机初始化模型（不加载权重）
model_random = unet_mod.UNet1D(
    obs_dim=config['obs_dim'],
    act_dim=config['act_dim'],
    base_channels=config['base_channels'],
    cond_dim=config['cond_dim'],
    time_dim=config['time_dim'],
).to(device)
model_random.eval()

# 加载训练好的模型
model_trained = unet_mod.UNet1D(
    obs_dim=config['obs_dim'],
    act_dim=config['act_dim'],
    base_channels=config['base_channels'],
    cond_dim=config['cond_dim'],
    time_dim=config['time_dim'],
).to(device)
model_trained.load_state_dict(ckpt['ema_state_dict'])
model_trained.eval()

print("检查模型 condition 注入")
print("=" * 60)
print()

# 固定输入
x = torch.randn(1, config['act_dim'], config['future']).to(device)  # (B, C, T)
t = torch.tensor([50], device=device)

print("测试 1: 随机初始化模型对 condition 的敏感性")
print("-" * 40)

cond1 = torch.zeros(1, config['obs_dim']).to(device)
cond2 = torch.randn(1, config['obs_dim']).to(device) * 10  # 差别很大的 condition

with torch.no_grad():
    out1 = model_random(x, t, cond1)
    out2 = model_random(x, t, cond2)

diff_random = (out1 - out2).abs().mean().item()
print(f"  cond1=zeros, cond2=randn*10")
print(f"  输出差异 (随机模型): {diff_random:.6f}")

print()
print("测试 2: 训练后模型对 condition 的敏感性")
print("-" * 40)

with torch.no_grad():
    out1 = model_trained(x, t, cond1)
    out2 = model_trained(x, t, cond2)

diff_trained = (out1 - out2).abs().mean().item()
print(f"  cond1=zeros, cond2=randn*10")
print(f"  输出差异 (训练后模型): {diff_trained:.6f}")

print()
print("测试 3: 检查 cond_proj 权重")
print("-" * 40)

# 检查 obs_proj 的权重
for name, param in model_trained.named_parameters():
    if 'obs_proj' in name or 'cond_proj' in name or 'cond_fuse' in name:
        print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, norm={param.norm().item():.6f}")

print()
print("测试 4: 训练前后权重对比")
print("-" * 40)

total_diff = 0.0
count = 0
for (name1, p1), (name2, p2) in zip(model_random.named_parameters(), model_trained.named_parameters()):
    diff = (p1 - p2).abs().mean().item()
    total_diff += diff
    count += 1
    
print(f"  平均参数差异: {total_diff / count:.6f}")

# 特别检查 cond 相关参数
print()
print("  condition 相关参数的差异:")
for (name1, p1), (name2, p2) in zip(model_random.named_parameters(), model_trained.named_parameters()):
    if 'obs_proj' in name1 or 'cond' in name1:
        diff = (p1 - p2).abs().mean().item()
        print(f"    {name1}: {diff:.6f}")

print()
print("测试 5: 梯度流检查")
print("-" * 40)

# 创建一个可训练的输入来检查梯度
x_grad = torch.randn(1, config['act_dim'], config['future'], device=device, requires_grad=True)
cond_grad = torch.randn(1, config['obs_dim'], device=device, requires_grad=True)
t_grad = torch.tensor([50], device=device)

out = model_trained(x_grad, t_grad, cond_grad)
loss = out.mean()
loss.backward()

print(f"  x 梯度 norm: {x_grad.grad.norm().item():.6f}")
print(f"  cond 梯度 norm: {cond_grad.grad.norm().item():.6f}")

if cond_grad.grad.norm().item() < 1e-6:
    print()
    print("  WARNING: condition 梯度接近 0，说明模型没有学会使用 condition！")
else:
    print()
    print("  OK: condition 有梯度流动")

print()
print("结论:")
print("=" * 60)
if diff_trained < 0.01:
    print("  训练后的模型对 condition 几乎不敏感（差异 < 0.01）")
    print("  这是模型坍塌的表现，需要调查原因")
else:
    print(f"  模型对 condition 敏感（差异 = {diff_trained:.4f}）")
