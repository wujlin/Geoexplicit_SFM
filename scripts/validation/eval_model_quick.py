"""Phase 4 模型快速评估脚本"""
import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/data"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/model"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/diffusion"))

from dataset import TrajectorySlidingWindow
from normalizer import ObsNormalizer, ActionNormalizer
from scheduler import DDIMScheduler
from unet1d import UNet1D


def load_model_and_data(ckpt_path):
    """加载模型和数据"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    
    model = UNet1D(
        obs_dim=config["obs_dim"],
        act_dim=config["act_dim"],
        base_channels=config["base_channels"],
        cond_dim=config["cond_dim"],
        time_dim=config["time_dim"],
    )
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()
    
    obs_norm = ObsNormalizer(mode="zscore", include_nav=True)
    obs_norm.load_state_dict(ckpt["obs_normalizer"])
    
    act_norm = ActionNormalizer(mode="zscore")
    act_norm.load_state_dict(ckpt["action_normalizer"])
    
    nav_data = np.load(PROJECT_ROOT / "data/processed/nav_baseline.npz")
    nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)
    
    ds = TrajectorySlidingWindow(
        h5_path=PROJECT_ROOT / "data/output/trajectories.h5",
        history=config["history"],
        future=config["future"],
        nav_field=nav_field,
        nav_fields_dir=PROJECT_ROOT / "data/processed/nav_fields",
    )
    
    return model, obs_norm, act_norm, ds, config


def eval_cos_sim(model, obs_norm, act_norm, ds, config, n_samples=100, seed=42):
    """评估 cos_sim(pred, GT)"""
    scheduler = DDIMScheduler(
        num_diffusion_steps=config["num_diffusion_steps"],
        num_inference_steps=10,
    )
    
    np.random.seed(seed)
    indices = np.random.choice(len(ds), n_samples, replace=False)
    
    cos_sims = []
    for idx in indices:
        sample = ds[int(idx)]
        obs = sample["obs"]
        action_gt = sample["action"]
        
        obs_normed = obs_norm.transform(obs.unsqueeze(0))
        global_cond = obs_normed.reshape(1, -1)
        
        torch.manual_seed(seed)
        x = torch.randn(1, config["act_dim"], config["future"])
        
        for t in scheduler.timesteps:
            t_batch = torch.full((1,), t, dtype=torch.long)
            with torch.no_grad():
                eps = model(x, t_batch, global_cond)
            x = scheduler.step(eps, t, x)
        
        pred = x.permute(0, 2, 1)[0, 0].numpy()
        gt = act_norm.transform(action_gt.unsqueeze(0))[0, 0].numpy()
        
        cos = np.dot(pred, gt) / (np.linalg.norm(pred) * np.linalg.norm(gt) + 1e-8)
        cos_sims.append(cos)
    
    return np.array(cos_sims)


def eval_condition_sensitivity(model, config):
    """评估 condition 敏感性"""
    from scheduler import DDPMScheduler
    scheduler = DDPMScheduler(num_diffusion_steps=100)
    
    torch.manual_seed(42)
    action = torch.randn(1, config["act_dim"], config["future"])
    noise = torch.randn_like(action)
    t = torch.tensor([50])
    noisy = scheduler.add_noise(action, noise, t)
    
    cond1 = torch.randn(1, config["obs_dim"])
    cond2 = torch.randn(1, config["obs_dim"])
    
    with torch.no_grad():
        pred1 = model(noisy, t, cond1)
        pred2 = model(noisy, t, cond2)
    
    diff_cond = (pred1 - pred2).abs().mean().item()
    
    noisy2 = scheduler.add_noise(-action, noise, t)
    with torch.no_grad():
        pred3 = model(noisy2, t, cond1)
    
    diff_input = (pred1 - pred3).abs().mean().item()
    
    return diff_cond, diff_input


if __name__ == "__main__":
    ckpt_path = PROJECT_ROOT / "data/output/phase4_checkpoints/best.pt"
    
    print("Loading model...")
    model, obs_norm, act_norm, ds, config = load_model_and_data(ckpt_path)
    
    print("\n=== cos_sim(pred, GT) ===")
    cos_sims = eval_cos_sim(model, obs_norm, act_norm, ds, config, n_samples=100)
    print(f"mean: {cos_sims.mean():.4f}")
    print(f">0.5: {(cos_sims > 0.5).mean()*100:.1f}%")
    print(f">0.8: {(cos_sims > 0.8).mean()*100:.1f}%")
    
    print("\n=== Condition Sensitivity ===")
    diff_cond, diff_input = eval_condition_sensitivity(model, config)
    print(f"diff from condition: {diff_cond:.4f}")
    print(f"diff from input: {diff_input:.4f}")
    print(f"ratio (cond/input): {diff_cond/diff_input:.2f}")
