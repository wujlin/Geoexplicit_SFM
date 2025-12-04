"""全面评估新训练的 CFG 模型

检查内容:
1. Checkpoint 配置（确认 CFG dropout 已启用）
2. 模型对 condition 的敏感性（对比旧模型）
3. 方向预测质量（cos_sim）
4. 不同 guidance_scale 的效果
"""

import sys
import numpy as np
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
import h5py
from pathlib import Path
import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

print("=" * 60)
print("Phase 4 新模型全面评估")
print("=" * 60)
print()

# ============================================================
# 1. 检查 Checkpoint 配置
# ============================================================
print("【1】Checkpoint 配置检查")
print("-" * 40)

phase4_root = Path('src/phase4')
normalizer_mod = _import_module('normalizer', phase4_root / 'data' / 'normalizer.py')
scheduler_mod = _import_module('scheduler', phase4_root / 'diffusion' / 'scheduler.py')
unet_mod = _import_module('unet1d', phase4_root / 'model' / 'unet1d.py')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('data/output/phase4_checkpoints/best.pt', map_location=device, weights_only=False)
config = ckpt['config']

print("模型配置:")
for k, v in config.items():
    print(f"  {k}: {v}")

print()
print(f"训练 epoch: {ckpt.get('epoch', 'N/A')}")
print(f"最终 loss: {ckpt.get('loss', 'N/A'):.6f}" if isinstance(ckpt.get('loss'), float) else f"最终 loss: {ckpt.get('loss', 'N/A')}")

cfg_dropout = config.get('cfg_dropout_prob', None)
if cfg_dropout is not None and cfg_dropout > 0:
    print(f"CFG dropout: {cfg_dropout} ✓")
else:
    print(f"WARNING: CFG dropout = {cfg_dropout} (未启用或为0)")

print()

# ============================================================
# 2. 加载模型
# ============================================================
print("【2】加载模型")
print("-" * 40)

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

print(f"模型加载成功，设备: {device}")
print()

# ============================================================
# 3. 测试 condition 敏感性
# ============================================================
print("【3】Condition 敏感性测试")
print("-" * 40)

cond1 = torch.zeros(1, config['obs_dim']).to(device)
cond2 = torch.randn(1, config['obs_dim']).to(device) * 5
shape = (1, config['future'], config['act_dim'])

print("不同 guidance_scale 下的 condition 敏感性:")
for scale in [1.0, 2.0, 3.0, 5.0]:
    gen1 = torch.Generator(device=device).manual_seed(42)
    gen2 = torch.Generator(device=device).manual_seed(42)
    
    with torch.no_grad():
        if scale > 1.0:
            pred1 = scheduler.sample_cfg(model, shape, cond1, device, guidance_scale=scale, generator=gen1)
            pred2 = scheduler.sample_cfg(model, shape, cond2, device, guidance_scale=scale, generator=gen2)
        else:
            pred1 = scheduler.sample(model, shape, cond1, device, generator=gen1)
            pred2 = scheduler.sample(model, shape, cond2, device, generator=gen2)
    
    diff = (pred1 - pred2).abs().mean().item()
    print(f"  scale={scale}: cond diff -> pred diff = {diff:.4f}")

print()

# ============================================================
# 4. 真实数据上的方向预测评估
# ============================================================
print("【4】真实数据方向预测评估")
print("-" * 40)

# 加载数据
h5_path = Path('data/output/trajectories.h5')
h5 = h5py.File(h5_path, 'r')
pos = h5['positions'][:]
vel = h5['velocities'][:]
dest = h5['destinations'][:]

# 加载导航场
nav_fields_dir = Path('data/processed/nav_fields')
nav_files = sorted(nav_fields_dir.glob('nav_field_*.npz'))
nav_data = {}
for f in nav_files:
    d = np.load(f)
    sink_id = int(f.stem.split('_')[2])
    nav_data[sink_id] = np.stack([d['nav_y'], d['nav_x']], axis=0)

H, W = 1365, 1435
history = config['history']
future = config['future']

def get_nav_at_pos(position, dest_id):
    y, x = position
    row = int(np.clip(y, 0, H - 1))
    col = int(np.clip(x, 0, W - 1))
    if dest_id in nav_data:
        return nav_data[dest_id][:, row, col]
    return np.zeros(2)

def build_obs(agent_idx, t0):
    obs_list = []
    for t in range(t0, t0 + history):
        p = pos[t, agent_idx]
        v = vel[t, agent_idx]
        d = dest[t, agent_idx]
        nav = get_nav_at_pos(p, d)
        obs_list.append(np.concatenate([p, v, nav]))
    return np.array(obs_list, dtype=np.float32)

def get_gt_action(agent_idx, t0):
    start_t = t0 + history
    end_t = start_t + future
    return vel[start_t:end_t, agent_idx].astype(np.float32)

# 收集有效样本
T, N = pos.shape[:2]
np.random.seed(42)
test_samples = []
n_samples = 200

for _ in range(50000):
    agent_idx = np.random.randint(N)
    t0 = np.random.randint(T - history - future)
    
    if dest[t0, agent_idx] == -1:
        continue
    if dest[t0 + history + future - 1, agent_idx] == -1:
        continue
    
    d0 = dest[t0, agent_idx]
    all_same_dest = all(dest[tt, agent_idx] == d0 for tt in range(t0, t0 + history + future))
    if not all_same_dest:
        continue
    
    # 检查 GT 是否在运动
    gt = get_gt_action(agent_idx, t0)
    gt_speed = np.linalg.norm(gt[0])
    if gt_speed < 0.3:  # 跳过静止样本
        continue
    
    test_samples.append((agent_idx, t0))
    if len(test_samples) >= n_samples:
        break

print(f"收集到 {len(test_samples)} 个有效测试样本")
print()

# 测试不同 guidance_scale
for guidance_scale in [1.0, 2.0, 3.0, 5.0]:
    results = []
    
    for agent_idx, t0 in test_samples:
        obs = build_obs(agent_idx, t0)
        gt = get_gt_action(agent_idx, t0)
        
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            obs_normed = obs_normalizer.transform(obs_tensor)
            global_cond = obs_normed.reshape(1, -1)
            
            generator = torch.Generator(device=device).manual_seed(42)
            shape = (1, future, config['act_dim'])
            
            if guidance_scale > 1.0:
                pred = scheduler.sample_cfg(model, shape, global_cond, device, 
                                           guidance_scale=guidance_scale, generator=generator)
            else:
                pred = scheduler.sample(model, shape, global_cond, device, generator=generator)
            
            pred = action_normalizer.inverse_transform(pred)
            pred = pred.cpu().numpy()[0]
        
        # 计算第一步的 cos_sim
        p, g = pred[0], gt[0]
        p_norm, g_norm = np.linalg.norm(p), np.linalg.norm(g)
        if p_norm > 1e-6 and g_norm > 1e-6:
            cos_sim = np.dot(p, g) / (p_norm * g_norm)
            results.append(cos_sim)
    
    results = np.array(results)
    print(f"guidance_scale={guidance_scale}:")
    print(f"  平均 cos_sim: {results.mean():.4f}")
    print(f"  cos_sim > 0: {(results > 0).mean()*100:.1f}%")
    print(f"  cos_sim > 0.5: {(results > 0.5).mean()*100:.1f}%")
    print(f"  cos_sim > 0.8: {(results > 0.8).mean()*100:.1f}%")
    print()

h5.close()

# ============================================================
# 5. 详细分析几个样本
# ============================================================
print("【5】样本详细分析")
print("-" * 40)

h5 = h5py.File(h5_path, 'r')
pos = h5['positions'][:]
vel = h5['velocities'][:]
dest = h5['destinations'][:]

print("查看前 5 个样本的预测 vs GT:")
for i, (agent_idx, t0) in enumerate(test_samples[:5]):
    obs = build_obs(agent_idx, t0)
    gt = get_gt_action(agent_idx, t0)
    
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    
    with torch.no_grad():
        obs_normed = obs_normalizer.transform(obs_tensor)
        global_cond = obs_normed.reshape(1, -1)
        generator = torch.Generator(device=device).manual_seed(42)
        shape = (1, future, config['act_dim'])
        
        # 使用 guidance_scale=3.0
        pred = scheduler.sample_cfg(model, shape, global_cond, device, 
                                   guidance_scale=3.0, generator=generator)
        pred = action_normalizer.inverse_transform(pred)
        pred = pred.cpu().numpy()[0]
    
    p, g = pred[0], gt[0]
    cos_sim = np.dot(p, g) / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-8)
    
    # 获取导航方向
    d = dest[t0, agent_idx]
    nav = get_nav_at_pos(pos[t0 + history - 1, agent_idx], d)
    
    print(f"样本 {i+1}: agent={agent_idx}, t0={t0}, dest={d}")
    print(f"  GT:   [{gt[0][0]:+.3f}, {gt[0][1]:+.3f}]")
    print(f"  Pred: [{pred[0][0]:+.3f}, {pred[0][1]:+.3f}]")
    print(f"  Nav:  [{nav[0]:+.3f}, {nav[1]:+.3f}]")
    print(f"  cos_sim(pred, gt): {cos_sim:.3f}")
    
    # 计算 pred 与 nav 的相似度
    nav_cos = np.dot(pred[0], nav) / (np.linalg.norm(pred[0]) * np.linalg.norm(nav) + 1e-8)
    gt_nav_cos = np.dot(gt[0], nav) / (np.linalg.norm(gt[0]) * np.linalg.norm(nav) + 1e-8)
    print(f"  cos_sim(pred, nav): {nav_cos:.3f}")
    print(f"  cos_sim(gt, nav): {gt_nav_cos:.3f}")
    print()

h5.close()

print("=" * 60)
print("评估完成")
print("=" * 60)
