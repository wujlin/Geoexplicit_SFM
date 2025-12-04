"""检查模型是否真的输出常数"""

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
obs_normalizer = normalizer_mod.ObsNormalizer()
obs_normalizer.load_state_dict(ckpt['obs_normalizer'])

scheduler = scheduler_mod.DDIMScheduler(
    num_diffusion_steps=config['num_diffusion_steps'],
    num_inference_steps=20,
    eta=0.0,
)

print("检查模型是否输出常数")
print("=" * 60)
print()

# 测试 1: 不同的 condition
print("测试 1: 不同的随机 condition")
print("-" * 40)

for i in range(5):
    cond = torch.randn(1, config['obs_dim']).to(device)
    
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(42)
        shape = (1, config['future'], config['act_dim'])
        pred = scheduler.sample(model, shape, cond, device, generator=generator)
    
    print(f"  cond mean={cond.mean().item():.3f}, pred[0]={pred[0, 0].cpu().numpy()}")

print()
print("测试 2: 相同 condition，不同初始噪声")
print("-" * 40)

cond = torch.randn(1, config['obs_dim']).to(device)
print(f"  固定 cond mean={cond.mean().item():.3f}")

for seed in range(5):
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(seed)
        shape = (1, config['future'], config['act_dim'])
        pred = scheduler.sample(model, shape, cond, device, generator=generator)
    
    print(f"  seed={seed}: pred[0]={pred[0, 0].cpu().numpy()}")

print()
print("测试 3: 检查归一化前后的原始输出")
print("-" * 40)

cond = torch.randn(1, config['obs_dim']).to(device)
with torch.no_grad():
    generator = torch.Generator(device=device).manual_seed(42)
    shape = (1, config['future'], config['act_dim'])
    pred_normed = scheduler.sample(model, shape, cond, device, generator=generator)
    pred_denormed = action_normalizer.inverse_transform(pred_normed)

print(f"  归一化空间输出: pred_normed[0]={pred_normed[0, 0].cpu().numpy()}")
print(f"  反归一化后: pred_denormed[0]={pred_denormed[0, 0].cpu().numpy()}")
print()

print("  action normalizer 参数:")
state = action_normalizer.state_dict()
for k, v in state.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {v.numpy()}")
    else:
        print(f"    {k}: {v}")

print()
print("测试 4: 模型单步预测")
print("-" * 40)
print("直接查看模型对不同输入的输出...")

# 固定噪声输入
x = torch.zeros(1, config['act_dim'], config['future']).to(device)  # (B, C, T)
t = torch.tensor([50], device=device)  # 中间时间步

for i in range(3):
    cond = torch.randn(1, config['obs_dim']).to(device)
    
    with torch.no_grad():
        out = model(x, t, cond)
    
    print(f"  cond mean={cond.mean().item():.3f}: model output mean={out.mean().item():.5f}, std={out.std().item():.5f}")

print()
print("测试 5: 条件敏感性测试")
print("-" * 40)

# 测试条件的变化是否影响输出
base_cond = torch.zeros(1, config['obs_dim']).to(device)

for scale in [0.0, 0.1, 1.0, 10.0]:
    cond = base_cond + torch.randn_like(base_cond) * scale
    
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(42)
        shape = (1, config['future'], config['act_dim'])
        pred = scheduler.sample(model, shape, cond, device, generator=generator)
    
    print(f"  cond scale={scale}: pred[0]={pred[0, 0].cpu().numpy()}")
