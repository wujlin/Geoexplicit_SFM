"""Phase 4 模型性能验证 - 使用固定种子进行公平测试

目标：
1. 使用固定种子确保可复现
2. 在多个样本上测试
3. 分析 cos_sim 分布
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

print("加载模型和数据...")
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
    eta=0.0,  # 确定性
)

# 使用预计算的有效索引文件避免内存问题
valid_indices_path = Path('data/processed/valid_indices.npy')
if valid_indices_path.exists():
    print("加载预计算的有效索引...")
    valid_indices = np.load(valid_indices_path)
else:
    print("预计算索引不存在，手动计算...")
    # 简单起见，我们直接从 h5 读取并手动选择样本
    valid_indices = None

# 加载 h5 数据
import h5py
h5_path = Path('data/output/trajectories.h5')
h5 = h5py.File(h5_path, 'r')
pos = h5['positions'][:]
vel = h5['velocities'][:]
dest = h5['destinations'][:]

# 加载 nav fields
nav_fields_dir = Path('data/processed/nav_fields')
nav_files = sorted(nav_fields_dir.glob('nav_field_*.npz'))
nav_data = {}
for f in nav_files:
    d = np.load(f)
    # 文件名格式: nav_field_000.npz
    sink_id = int(f.stem.split('_')[2])
    nav_data[sink_id] = np.stack([d['nav_y'], d['nav_x']], axis=0)  # (2, H, W)

H, W = 1365, 1435  # 栅格大小

def get_nav_at_pos(position, dest_id):
    """获取某位置的导航向量"""
    y, x = position
    row = int(np.clip(y, 0, H - 1))
    col = int(np.clip(x, 0, W - 1))
    nav = nav_data[dest_id][:, row, col]
    return nav

def build_obs(agent_idx, t0, history=2):
    """构建 observation"""
    obs_list = []
    for t in range(t0, t0 + history):
        p = pos[t, agent_idx]
        v = vel[t, agent_idx]
        d = dest[t, agent_idx]
        nav = get_nav_at_pos(p, d)
        obs_list.append(np.concatenate([p, v, nav]))
    return np.array(obs_list, dtype=np.float32)  # (history, 6)

def get_gt_action(agent_idx, t0, history=2, future=8):
    """获取 ground truth action"""
    start_t = t0 + history
    end_t = start_t + future
    return vel[start_t:end_t, agent_idx].astype(np.float32)  # (future, 2)

print()
print("=" * 60)
print("Phase 4 模型性能验证")
print("=" * 60)
print()

# 选择一些有效样本进行测试
# 条件：非 padding (dest != -1) 且有足够的未来步
T, N = pos.shape[:2]
history = config['history']
future = config['future']

# 找有效样本
np.random.seed(42)
test_samples = []
max_try = 10000
n_samples = 100

for _ in range(max_try):
    agent_idx = np.random.randint(N)
    t0 = np.random.randint(T - history - future)
    
    # 检查 dest 是否有效
    if dest[t0, agent_idx] == -1:
        continue
    if dest[t0 + history + future - 1, agent_idx] == -1:
        continue
    
    # 检查 dest 是否一致（排除 dest 变化的情况）
    d0 = dest[t0, agent_idx]
    all_same_dest = True
    for tt in range(t0, t0 + history + future):
        if dest[tt, agent_idx] != d0:
            all_same_dest = False
            break
    if not all_same_dest:
        continue
    
    test_samples.append((agent_idx, t0))
    if len(test_samples) >= n_samples:
        break

print(f"收集到 {len(test_samples)} 个有效测试样本")
print()

# 运行推理
results = []

for i, (agent_idx, t0) in enumerate(test_samples):
    # 构建数据
    obs = build_obs(agent_idx, t0, history)
    gt = get_gt_action(agent_idx, t0, history, future)
    
    # 转为 tensor
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1, history, 6)
    
    # 归一化和推理
    with torch.no_grad():
        obs_normed = obs_normalizer.transform(obs_tensor)
        global_cond = obs_normed.reshape(1, -1)
        
        # 使用固定种子
        generator = torch.Generator(device=device).manual_seed(42)
        shape = (1, future, config['act_dim'])
        pred = scheduler.sample(model, shape, global_cond, device, generator=generator)
        pred = action_normalizer.inverse_transform(pred)
        pred = pred.cpu().numpy()[0]
    
    # 计算 cos_sim (第一步)
    p, g = pred[0], gt[0]
    p_norm, g_norm = np.linalg.norm(p), np.linalg.norm(g)
    if p_norm > 1e-6 and g_norm > 1e-6:
        cos_sim = np.dot(p, g) / (p_norm * g_norm)
    else:
        cos_sim = 0.0
    
    # GT 方向一致性
    gt_consistency = []
    for j in range(future - 1):
        g1, g2 = gt[j], gt[j + 1]
        n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
        if n1 > 0.1 and n2 > 0.1:
            gt_consistency.append(np.dot(g1, g2) / (n1 * n2))
    avg_gt_consistency = np.mean(gt_consistency) if gt_consistency else 0.0
    
    results.append({
        'agent': agent_idx,
        't0': t0,
        'cos_sim': cos_sim,
        'gt_consistency': avg_gt_consistency,
        'pred': pred[0],
        'gt': gt[0],
    })

h5.close()

# 分析结果
cos_sims = [r['cos_sim'] for r in results]
gt_consistencies = [r['gt_consistency'] for r in results]

print("整体结果:")
print(f"  平均 cos_sim: {np.mean(cos_sims):.4f}")
print(f"  cos_sim 标准差: {np.std(cos_sims):.4f}")
print(f"  cos_sim > 0 的比例: {np.mean(np.array(cos_sims) > 0):.1%}")
print(f"  cos_sim > 0.5 的比例: {np.mean(np.array(cos_sims) > 0.5):.1%}")
print()

print("按 GT 一致性分组:")
high_consistency = [r for r in results if r['gt_consistency'] > 0.5]
low_consistency = [r for r in results if r['gt_consistency'] < 0.5]

if high_consistency:
    print(f"  高一致性 (GT cons > 0.5): {len(high_consistency)} 样本")
    print(f"    平均 cos_sim: {np.mean([r['cos_sim'] for r in high_consistency]):.4f}")
    
if low_consistency:
    print(f"  低一致性 (GT cons < 0.5): {len(low_consistency)} 样本")
    print(f"    平均 cos_sim: {np.mean([r['cos_sim'] for r in low_consistency]):.4f}")

print()
print("前 10 个样本详情:")
for r in results[:10]:
    print(f"  agent={r['agent']}, t0={r['t0']}: cos_sim={r['cos_sim']:.3f}, gt_cons={r['gt_consistency']:.3f}")
    print(f"    pred={r['pred']}, gt={r['gt']}")
