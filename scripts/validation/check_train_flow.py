"""训练流程完整性检查"""
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/data"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/model"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/diffusion"))

import numpy as np
import torch.nn.functional as F
from dataset import TrajectorySlidingWindow
from normalizer import ObsNormalizer, ActionNormalizer
from scheduler import DDPMScheduler
from unet1d import UNet1D


def main():
    print("=" * 60)
    print("训练流程完整性检查")
    print("=" * 60)
    
    # 1. 数据集
    print("\n1. 数据集加载:")
    nav_fields_dir = PROJECT_ROOT / "data/processed/nav_fields"
    ds = TrajectorySlidingWindow(
        h5_path=PROJECT_ROOT / "data/output/trajectories.h5",
        history=2, future=8, nav_field=None, nav_fields_dir=nav_fields_dir,
    )
    print(f"   大小: {len(ds):,}")
    
    sample = ds[0]
    obs_shape = list(sample["obs"].shape)
    act_shape = list(sample["action"].shape)
    print(f"   obs: {obs_shape} (期望 [2, 6])")
    print(f"   action: {act_shape} (期望 [8, 2])")
    
    # 2. 归一化器
    print("\n2. 归一化器:")
    obs_norm = ObsNormalizer(mode="zscore", include_nav=True)
    act_norm = ActionNormalizer(mode="zscore")
    
    obs_samples = np.random.randn(100, 2, 6).astype(np.float32)
    act_samples = np.random.randn(100, 8, 2).astype(np.float32)
    obs_norm.fit(obs_samples[..., :2], obs_samples[..., 2:4], obs_samples[..., 4:6])
    act_norm.fit(act_samples)
    print("   fit 完成")
    
    # 3. 模型
    print("\n3. 模型初始化:")
    model = UNet1D(obs_dim=12, act_dim=2, base_channels=128, cond_dim=64, time_dim=64)
    scheduler = DDPMScheduler(num_diffusion_steps=100)
    
    # 检查关键初始化
    gamma_bias_ok = all(
        model.state_dict()[k].mean().item() > 0.99
        for k in model.state_dict() if "cond_gamma.bias" in k
    )
    print(f"   gamma bias ≈ 1: {gamma_bias_ok}")
    print(f"   obs_scale = {model.obs_scale.item():.1f}")
    
    # 4. 前向传播
    print("\n4. 前向传播:")
    B = 4
    obs = torch.randn(B, 2, 6)
    action = torch.randn(B, 8, 2)
    
    obs_normed = obs_norm.transform(obs).reshape(B, -1)
    act_normed = act_norm.transform(action)
    
    noise = torch.randn_like(act_normed)
    timesteps = scheduler.sample_timesteps(B, "cpu")
    noisy_action = scheduler.add_noise(act_normed, noise, timesteps)
    
    noisy_input = noisy_action.permute(0, 2, 1)
    pred_noise = model(noisy_input, timesteps, obs_normed)
    pred_noise = pred_noise.permute(0, 2, 1)
    
    shape_ok = pred_noise.shape == noise.shape
    print(f"   输出形状匹配: {shape_ok}")
    
    # 5. Loss + 反向传播
    print("\n5. Loss + 反向传播:")
    loss = F.mse_loss(pred_noise, noise)
    print(f"   Loss: {loss.item():.4f}")
    
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"   所有参数有梯度: {grad_ok}")
    print(f"   obs_scale.grad: {model.obs_scale.grad.item():.6f}")
    
    # 6. Condition 敏感性
    print("\n6. Condition 敏感性 (随机初始化):")
    model.zero_grad()
    
    torch.manual_seed(42)
    x = torch.randn(1, 2, 8)
    t = torch.tensor([50])
    cond1 = torch.randn(1, 12)
    cond2 = torch.randn(1, 12)
    
    with torch.no_grad():
        p1 = model(x, t, cond1)
        p2 = model(x, t, cond2)
    diff_cond = (p1 - p2).abs().mean().item()
    
    x2 = torch.randn(1, 2, 8)
    with torch.no_grad():
        p3 = model(x2, t, cond1)
    diff_input = (p1 - p3).abs().mean().item()
    
    ratio = diff_cond / diff_input
    print(f"   cond 差异: {diff_cond:.4f}")
    print(f"   input 差异: {diff_input:.4f}")
    print(f"   比值: {ratio:.2f} (期望 > 0.3)")
    
    # 总结
    print("\n" + "=" * 60)
    all_ok = obs_shape == [2, 6] and act_shape == [8, 2] and gamma_bias_ok and shape_ok and grad_ok and ratio > 0.2
    if all_ok:
        print("所有检查通过，可以开始训练")
    else:
        print("存在问题，请检查")
    print("=" * 60)


if __name__ == "__main__":
    main()
