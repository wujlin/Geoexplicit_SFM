"""
诊断 Diffusion Policy 推理问题
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE4_ROOT = PROJECT_ROOT / "src" / "phase4"

# 动态导入
import importlib.util

def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config = _import_module("config", PHASE4_ROOT / "config.py")
normalizer_module = _import_module("normalizer", PHASE4_ROOT / "data" / "normalizer.py")
scheduler_module = _import_module("scheduler", PHASE4_ROOT / "diffusion" / "scheduler.py")
unet_module = _import_module("unet1d", PHASE4_ROOT / "model" / "unet1d.py")

ActionNormalizer = normalizer_module.ActionNormalizer
DDIMScheduler = scheduler_module.DDIMScheduler
UNet1D = unet_module.UNet1D


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载检查点
    checkpoint_path = PROJECT_ROOT / "data" / "output" / "phase4_checkpoints" / "best.pt"
    
    # numpy 兼容性
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        sys.modules['numpy._core.numeric'] = np.core.numeric
        print("Applied numpy compatibility patch")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print("\n=== 检查点信息 ===")
    print(f"Config: {checkpoint['config']}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Loss: {checkpoint.get('loss', 'N/A')}")
    
    print("\n=== Action Normalizer ===")
    normalizer_state = checkpoint['action_normalizer']
    print(f"Mode: {normalizer_state.get('mode', 'N/A')}")
    norm_inner = normalizer_state.get('normalizer', {})
    print(f"Min: {norm_inner.get('min_val', 'N/A')}")
    print(f"Max: {norm_inner.get('max_val', 'N/A')}")
    
    # 加载模型
    cfg = checkpoint['config']
    model = UNet1D(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        base_channels=cfg['base_channels'],
        cond_dim=cfg['cond_dim'],
        time_dim=cfg['time_dim'],
    ).to(device)
    
    model.load_state_dict(checkpoint['ema_state_dict'])
    model.eval()
    
    # 加载归一化器
    action_normalizer = ActionNormalizer()
    action_normalizer.load_state_dict(checkpoint['action_normalizer'])
    
    # 创建调度器
    scheduler = DDIMScheduler(
        num_diffusion_steps=cfg['num_diffusion_steps'],
        num_inference_steps=20,
    )
    
    print("\n=== 测试推理 ===")
    
    # 创建一个假的观测
    # obs: (history, 4) = (2, 4) -> [pos_y, pos_x, vel_y, vel_x]
    history = cfg['history']
    obs = np.array([
        [500.0, 500.0, 0.5, 0.5],  # t-1
        [500.5, 500.5, 0.5, 0.5],  # t
    ], dtype=np.float32)
    
    print(f"Input obs shape: {obs.shape}")
    print(f"Input obs:\n{obs}")
    
    # 转换为条件
    obs_tensor = torch.from_numpy(obs.reshape(1, -1)).to(device)
    print(f"Condition shape: {obs_tensor.shape}")
    
    # 采样
    shape = (1, cfg['future'], cfg['act_dim'])
    print(f"Sampling shape: {shape}")
    
    with torch.no_grad():
        samples = scheduler.sample(
            model=model,
            shape=shape,
            condition=obs_tensor,
            device=device,
        )
    
    print(f"\n=== 采样结果 (归一化后) ===")
    print(f"Shape: {samples.shape}")
    print(f"Min: {samples.min().item():.4f}, Max: {samples.max().item():.4f}")
    print(f"Mean: {samples.mean().item():.4f}, Std: {samples.std().item():.4f}")
    print(f"Sample:\n{samples[0].cpu().numpy()}")
    
    # 反归一化
    samples_denorm = action_normalizer.inverse_transform(samples)
    
    print(f"\n=== 反归一化后 ===")
    print(f"Min: {samples_denorm.min().item():.4f}, Max: {samples_denorm.max().item():.4f}")
    print(f"Mean: {samples_denorm.mean().item():.4f}, Std: {samples_denorm.std().item():.4f}")
    print(f"Sample:\n{samples_denorm[0].cpu().numpy()}")
    
    # 计算位移
    actions = samples_denorm[0].cpu().numpy()
    print(f"\n=== 位移分析 ===")
    for i, action in enumerate(actions):
        displacement = np.linalg.norm(action)
        print(f"Step {i+1}: vel=({action[0]:.4f}, {action[1]:.4f}), |v|={displacement:.4f}")


if __name__ == "__main__":
    main()
