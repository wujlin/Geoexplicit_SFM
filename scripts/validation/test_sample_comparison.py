"""对比不同样本的模型性能"""

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
)

dataset = dataset_mod.TrajectorySlidingWindow(
    h5_path='data/output/trajectories.h5',
    history=config['history'],
    future=config['future'],
    nav_fields_dir='data/processed/nav_fields',
)

def test_sample(idx):
    sample = dataset[idx]
    obs = sample['obs'].unsqueeze(0).to(device)
    gt_action = sample['action'].numpy()
    
    with torch.no_grad():
        obs_normed = obs_normalizer.transform(obs)
        global_cond = obs_normed.reshape(1, -1)
        
        shape = (1, config['future'], config['act_dim'])
        pred = scheduler.sample(model, shape, global_cond, device)
        pred = action_normalizer.inverse_transform(pred)
        pred = pred.cpu().numpy()[0]
    
    cos_sims = []
    for i in range(config['future']):
        p, g = pred[i], gt_action[i]
        p_norm, g_norm = np.linalg.norm(p), np.linalg.norm(g)
        if p_norm > 1e-6 and g_norm > 1e-6:
            cos = np.dot(p, g) / (p_norm * g_norm)
            cos_sims.append(cos)
    
    return np.mean(cos_sims), pred, gt_action, sample

print("对比 idx=1000 和随机样本")
print("=" * 60)

# idx=1000
cos_sim, pred, gt, sample = test_sample(1000)
print(f"idx=1000: cos_sim={cos_sim:.4f}")
print(f"  agent={sample['agent']}, t0={sample['t0']}, dest={sample['dest']}")
print(f"  pred[0]={pred[0]}, gt[0]={gt[0]}")
print()

# 随机样本
np.random.seed(42)
for idx in np.random.choice(len(dataset), 10, replace=False):
    cos_sim, pred, gt, sample = test_sample(idx)
    print(f"idx={idx}: cos_sim={cos_sim:.4f}")
    print(f"  agent={sample['agent']}, t0={sample['t0']}, dest={sample['dest']}")
    print(f"  pred[0]={pred[0]}, gt[0]={gt[0]}")
    
    # 检查 GT 是否在震荡
    gt_dirs = []
    for i in range(len(gt)-1):
        g1, g2 = gt[i], gt[i+1]
        n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
        if n1 > 0.1 and n2 > 0.1:
            gt_dirs.append(np.dot(g1, g2) / (n1 * n2))
    if gt_dirs:
        print(f"  GT 方向一致性: {np.mean(gt_dirs):.3f}")
    print()
