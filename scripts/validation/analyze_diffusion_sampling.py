"""深入分析扩散采样过程"""

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

# 手动创建 DDIM scheduler
scheduler = scheduler_mod.DDIMScheduler(
    num_diffusion_steps=config['num_diffusion_steps'],
    num_inference_steps=20,
    eta=0.0,
)

print("深入分析扩散采样过程")
print("=" * 60)
print()

# 固定 condition
cond = torch.zeros(1, config['obs_dim']).to(device)

# 固定初始噪声
generator = torch.Generator(device=device).manual_seed(42)
shape = (1, config['future'], config['act_dim'])
initial_noise = torch.randn(shape, device=device, generator=generator)

print(f"初始噪声统计: mean={initial_noise.mean():.4f}, std={initial_noise.std():.4f}")
print(f"初始噪声 [0,0]: {initial_noise[0, 0].cpu().numpy()}")
print()

# 手动单步采样，查看每步的变化
sample = initial_noise.clone()
timesteps = scheduler.timesteps.to(device)

print("DDIM 采样过程跟踪:")
print("-" * 40)

for i, t in enumerate(timesteps[:5]):  # 只看前 5 步
    t_batch = torch.full((1,), t, device=device, dtype=torch.long)
    
    # 获取上一个时间步
    if i + 1 < len(timesteps):
        prev_t = timesteps[i + 1]
    else:
        prev_t = torch.tensor(0)
    prev_t_batch = torch.full((1,), prev_t, device=device, dtype=torch.long)
    
    # 模型预测
    sample_input = sample.permute(0, 2, 1)  # (1, 8, 2) -> (1, 2, 8)
    with torch.no_grad():
        model_output = model(sample_input, t_batch, cond)
    model_output = model_output.permute(0, 2, 1)  # (1, 2, 8) -> (1, 8, 2)
    
    print(f"Step {i+1} (t={t}):")
    print(f"  sample before: mean={sample.mean():.4f}, std={sample.std():.4f}")
    print(f"  model output (pred noise): mean={model_output.mean():.4f}, std={model_output.std():.4f}")
    print(f"  model output [0,0]: {model_output[0, 0].cpu().numpy()}")
    
    # 去噪
    sample = scheduler.step(
        model_output, t_batch, sample,
        prev_timestep=prev_t_batch,
        generator=generator
    )
    
    print(f"  sample after: mean={sample.mean():.4f}, std={sample.std():.4f}")
    print()

print("最终输出:")
print(f"  归一化空间: {sample[0, 0].cpu().numpy()}")

# 反归一化
final_denormed = action_normalizer.inverse_transform(sample)
print(f"  反归一化后: {final_denormed[0, 0].cpu().numpy()}")

print()
print("=" * 60)
print("对比：不同 condition 的模型输出")
print("-" * 40)

for i in range(3):
    cond_test = torch.randn(1, config['obs_dim']).to(device) * 3
    
    # 使用相同初始噪声
    sample_test = initial_noise.clone()
    t_batch = torch.full((1,), timesteps[0], device=device, dtype=torch.long)
    
    sample_input = sample_test.permute(0, 2, 1)
    with torch.no_grad():
        model_output = model(sample_input, t_batch, cond_test)
    
    print(f"cond {i} (mean={cond_test.mean():.2f}): model output mean={model_output.mean():.5f}, std={model_output.std():.5f}")

print()
print("=" * 60)
print("关键问题：模型预测的噪声是否合理？")
print("-" * 40)

# 检查模型预测的噪声是否接近标准高斯
x_noise = torch.randn(1, config['act_dim'], config['future']).to(device)
t_mid = torch.tensor([50], device=device)
cond_test = torch.zeros(1, config['obs_dim']).to(device)

with torch.no_grad():
    pred_noise = model(x_noise, t_mid, cond_test)

print(f"输入噪声: mean={x_noise.mean():.4f}, std={x_noise.std():.4f}")
print(f"预测噪声: mean={pred_noise.mean():.4f}, std={pred_noise.std():.4f}")
print()

# 正常情况下，预测噪声应该接近输入噪声（因为目标是预测加入的噪声）
# 如果预测噪声总是某个固定值，说明模型没有学会
print("检查预测噪声的变化性:")
for seed in range(3):
    torch.manual_seed(seed)
    x_noise = torch.randn(1, config['act_dim'], config['future']).to(device)
    
    with torch.no_grad():
        pred_noise = model(x_noise, t_mid, cond_test)
    
    print(f"  seed={seed}: input_mean={x_noise.mean():.4f}, pred_mean={pred_noise.mean():.4f}, pred_std={pred_noise.std():.4f}")
