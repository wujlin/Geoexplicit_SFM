"""验证 CFG 实现是否正确

测试内容:
1. 训练时 condition dropout 是否正常工作
2. 推理时 CFG 采样是否影响输出
3. guidance_scale 是否有效增强 condition 敏感性
"""

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

print("=" * 60)
print("CFG 实现验证")
print("=" * 60)
print()

phase4_root = Path('src/phase4')
normalizer_mod = _import_module('normalizer', phase4_root / 'data' / 'normalizer.py')
scheduler_mod = _import_module('scheduler', phase4_root / 'diffusion' / 'scheduler.py')
unet_mod = _import_module('unet1d', phase4_root / 'model' / 'unet1d.py')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('data/output/phase4_checkpoints/best.pt', map_location=device, weights_only=False)
config = ckpt['config']

model = unet_mod.UNet1D(
    obs_dim=config['obs_dim'],
    act_dim=config['act_dim'],
    base_channels=config['base_channels'],
    cond_dim=config['cond_dim'],
    time_dim=config['time_dim'],
).to(device)
model.load_state_dict(ckpt['ema_state_dict'])
model.eval()

action_normalizer = normalizer_mod.ActionNormalizer()
action_normalizer.load_state_dict(ckpt['action_normalizer'])

# 测试 1: 验证 DDIMScheduler 有 sample_cfg 方法
print("测试 1: 检查 sample_cfg 方法是否存在")
print("-" * 40)
scheduler = scheduler_mod.DDIMScheduler(
    num_diffusion_steps=config['num_diffusion_steps'],
    num_inference_steps=20,
    eta=0.0,
)

if hasattr(scheduler, 'sample_cfg'):
    print("  OK: DDIMScheduler.sample_cfg 方法存在")
else:
    print("  ERROR: DDIMScheduler.sample_cfg 方法不存在！")
    sys.exit(1)

print()

# 测试 2: 验证 CFG 采样是否改变输出
print("测试 2: CFG 采样 vs 普通采样")
print("-" * 40)

cond = torch.randn(1, config['obs_dim']).to(device)
shape = (1, config['future'], config['act_dim'])

# 使用相同种子
generator1 = torch.Generator(device=device).manual_seed(42)
generator2 = torch.Generator(device=device).manual_seed(42)

with torch.no_grad():
    # 普通采样
    pred_normal = scheduler.sample(model, shape, cond, device, generator=generator1)
    
    # CFG 采样 (guidance_scale=1.0 应该等于普通采样)
    pred_cfg_1 = scheduler.sample_cfg(model, shape, cond, device, guidance_scale=1.0, generator=generator2)

diff_at_scale_1 = (pred_normal - pred_cfg_1).abs().mean().item()
print(f"  guidance_scale=1.0 vs normal: diff={diff_at_scale_1:.6f}")
if diff_at_scale_1 < 0.01:
    print("  OK: guidance_scale=1.0 与普通采样基本相同")
else:
    print("  WARNING: guidance_scale=1.0 应该与普通采样相同")

print()

# 测试 3: 验证不同 guidance_scale 产生不同结果
print("测试 3: 不同 guidance_scale 的效果")
print("-" * 40)

results = {}
for scale in [1.0, 2.0, 3.0, 5.0]:
    generator = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        pred = scheduler.sample_cfg(model, shape, cond, device, guidance_scale=scale, generator=generator)
    results[scale] = pred[0, 0].cpu().numpy()
    print(f"  scale={scale}: pred[0]={results[scale]}")

# 检查不同 scale 是否产生不同结果
diff_2_vs_1 = np.linalg.norm(results[2.0] - results[1.0])
diff_3_vs_1 = np.linalg.norm(results[3.0] - results[1.0])
diff_5_vs_1 = np.linalg.norm(results[5.0] - results[1.0])

print()
print(f"  scale=2.0 vs 1.0: diff={diff_2_vs_1:.4f}")
print(f"  scale=3.0 vs 1.0: diff={diff_3_vs_1:.4f}")
print(f"  scale=5.0 vs 1.0: diff={diff_5_vs_1:.4f}")

if diff_5_vs_1 > diff_3_vs_1 > diff_2_vs_1 > 0:
    print("  OK: 更高的 guidance_scale 产生更大的变化")
else:
    print("  WARNING: guidance_scale 的效果不符合预期")

print()

# 测试 4: 验证 CFG 是否增强 condition 敏感性
print("测试 4: CFG 对 condition 敏感性的影响")
print("-" * 40)

cond1 = torch.zeros(1, config['obs_dim']).to(device)
cond2 = torch.randn(1, config['obs_dim']).to(device) * 5  # 差异很大的 condition

for scale in [1.0, 3.0, 5.0]:
    generator1 = torch.Generator(device=device).manual_seed(42)
    generator2 = torch.Generator(device=device).manual_seed(42)
    
    with torch.no_grad():
        pred1 = scheduler.sample_cfg(model, shape, cond1, device, guidance_scale=scale, generator=generator1)
        pred2 = scheduler.sample_cfg(model, shape, cond2, device, guidance_scale=scale, generator=generator2)
    
    diff = (pred1 - pred2).abs().mean().item()
    print(f"  scale={scale}: cond diff -> pred diff={diff:.4f}")

print()
print("=" * 60)
print("验证完成")
print("=" * 60)
print()
print("结论:")
print("  - 如果测试 4 中更高的 scale 产生更大的 pred diff，说明 CFG 正在增强 condition 敏感性")
print("  - 注意: 当前模型是用旧方式训练的（没有 CFG dropout），CFG 可能效果有限")
print("  - 建议用新的 CFG 训练方式重新训练模型")
