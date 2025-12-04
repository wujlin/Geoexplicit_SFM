"""测试随机种子对 DDIM 采样的影响"""

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
dataset_mod = _import_module('dataset', phase4_root / 'data' / 'dataset.py')

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
    eta=0.0,  # 确定性采样
)

# 手动创建一个简单的 obs
obs = torch.randn(1, config['history'], 6).to(device)
obs_normed = obs_normalizer.transform(obs)
global_cond = obs_normed.reshape(1, -1)

print("测试 1: 不同随机种子的结果")
print("=" * 60)

for seed in [0, 42, 123]:
    # 方法1: 设置全局种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    with torch.no_grad():
        shape = (1, config['future'], config['act_dim'])
        pred = scheduler.sample(model, shape, global_cond, device, generator=None)
    
    print(f"seed={seed} (global seed): pred[0]={pred[0, 0].cpu().numpy()}")

print()
print("测试 2: 使用 generator 的结果")
print("=" * 60)

for seed in [0, 42, 123]:
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with torch.no_grad():
        shape = (1, config['future'], config['act_dim'])
        pred = scheduler.sample(model, shape, global_cond, device, generator=generator)
    
    print(f"seed={seed} (generator): pred[0]={pred[0, 0].cpu().numpy()}")

print()
print("测试 3: 同一种子多次运行")
print("=" * 60)

for trial in range(3):
    generator = torch.Generator(device=device).manual_seed(42)
    
    with torch.no_grad():
        shape = (1, config['future'], config['act_dim'])
        pred = scheduler.sample(model, shape, global_cond, device, generator=generator)
    
    print(f"trial {trial+1}: pred[0]={pred[0, 0].cpu().numpy()}")
