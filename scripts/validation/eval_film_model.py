"""FiLM 模型评估脚本 - 对照 CODE_DATA_STRUCTURE.md"""
import numpy as np
import torch
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/data"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/model"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/diffusion"))

from dataset import TrajectorySlidingWindow
from normalizer import ObsNormalizer, ActionNormalizer
from scheduler import DDIMScheduler, DDPMScheduler
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
    
    # 加载个体导航场
    nav_fields_dir = PROJECT_ROOT / "data/processed/nav_fields"
    
    ds = TrajectorySlidingWindow(
        h5_path=PROJECT_ROOT / "data/output/trajectories.h5",
        history=config["history"],
        future=config["future"],
        nav_field=None,
        nav_fields_dir=nav_fields_dir,
    )
    
    return model, obs_norm, act_norm, ds, config


def eval_condition_sensitivity(model, config):
    """评估 condition 敏感性 (对照文档 10.2)"""
    scheduler = DDPMScheduler(num_diffusion_steps=100)
    
    torch.manual_seed(42)
    action = torch.randn(1, config["act_dim"], config["future"])
    noise = torch.randn_like(action)
    t = torch.tensor([50])
    noisy = scheduler.add_noise(action, noise, t)
    
    # 测试 1: 相同 x_t + 不同 condition
    cond1 = torch.randn(1, config["obs_dim"])
    cond2 = torch.randn(1, config["obs_dim"])
    
    with torch.no_grad():
        pred1 = model(noisy, t, cond1)
        pred2 = model(noisy, t, cond2)
    
    diff_cond = (pred1 - pred2).abs().mean().item()
    
    # 测试 2: 不同 x_t + 相同 condition
    noisy2 = scheduler.add_noise(-action, noise, t)
    with torch.no_grad():
        pred3 = model(noisy2, t, cond1)
    
    diff_input = (pred1 - pred3).abs().mean().item()
    
    return diff_cond, diff_input


def eval_cos_sim_with_gt(model, obs_norm, act_norm, ds, config, n_samples=200, seed=42):
    """评估 cos_sim(pred, GT) - 使用个体导航场 (对照文档 9.3)"""
    scheduler = DDIMScheduler(
        num_diffusion_steps=config["num_diffusion_steps"],
        num_inference_steps=10,
    )
    
    np.random.seed(seed)
    indices = np.random.choice(len(ds), n_samples, replace=False)
    
    results = {"cos_pred_gt": [], "cos_pred_nav": [], "cos_gt_nav": []}
    
    for idx in indices:
        sample = ds[int(idx)]
        obs = sample["obs"]           # (history, 6)
        action_gt = sample["action"]  # (future, 2)
        
        # 归一化 obs
        obs_normed = obs_norm.transform(obs.unsqueeze(0))
        global_cond = obs_normed.reshape(1, -1)
        
        # DDIM 采样
        torch.manual_seed(seed)
        x = torch.randn(1, config["act_dim"], config["future"])
        
        for t in scheduler.timesteps:
            t_batch = torch.full((1,), t, dtype=torch.long)
            with torch.no_grad():
                eps = model(x, t_batch, global_cond)
            x = scheduler.step(eps, t, x)
        
        # 取第一步预测
        pred = x.permute(0, 2, 1)[0, 0].numpy()  # (2,)
        
        # GT (归一化空间)
        gt_normed = act_norm.transform(action_gt.unsqueeze(0))
        gt = gt_normed[0, 0].numpy()  # (2,)
        
        # nav direction (从 obs 的最后一帧提取)
        nav = obs[-1, 4:6].numpy()  # (2,) [nav_y, nav_x]
        
        # 计算 cos_sim
        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
        results["cos_pred_gt"].append(cos_sim(pred, gt))
        results["cos_pred_nav"].append(cos_sim(pred, nav))
        results["cos_gt_nav"].append(cos_sim(gt, nav))
    
    return {k: np.array(v) for k, v in results.items()}


def eval_output_diversity(model, obs_norm, ds, config, n_samples=50):
    """评估模型输出多样性 (不同 condition 应该有不同输出)"""
    scheduler = DDIMScheduler(
        num_diffusion_steps=config["num_diffusion_steps"],
        num_inference_steps=10,
    )
    
    np.random.seed(42)
    indices = np.random.choice(len(ds), n_samples, replace=False)
    
    outputs = []
    for idx in indices:
        sample = ds[int(idx)]
        obs = sample["obs"]
        
        obs_normed = obs_norm.transform(obs.unsqueeze(0))
        global_cond = obs_normed.reshape(1, -1)
        
        torch.manual_seed(123)  # 固定噪声种子
        x = torch.randn(1, config["act_dim"], config["future"])
        
        for t in scheduler.timesteps:
            t_batch = torch.full((1,), t, dtype=torch.long)
            with torch.no_grad():
                eps = model(x, t_batch, global_cond)
            x = scheduler.step(eps, t, x)
        
        pred = x.permute(0, 2, 1)[0, 0].numpy()
        outputs.append(pred)
    
    outputs = np.array(outputs)  # (n_samples, 2)
    return outputs


def eval_cfg_effect(model, obs_norm, ds, config, guidance_scales=[1.0, 3.0, 5.0]):
    """评估 CFG guidance_scale 效果"""
    scheduler = DDIMScheduler(
        num_diffusion_steps=config["num_diffusion_steps"],
        num_inference_steps=10,
    )
    
    np.random.seed(42)
    sample = ds[np.random.randint(len(ds))]
    obs = sample["obs"]
    
    obs_normed = obs_norm.transform(obs.unsqueeze(0))
    cond = obs_normed.reshape(1, -1)
    uncond = torch.zeros_like(cond)
    
    results = {}
    for scale in guidance_scales:
        torch.manual_seed(123)
        x = torch.randn(1, config["act_dim"], config["future"])
        
        for t in scheduler.timesteps:
            t_batch = torch.full((1,), t, dtype=torch.long)
            with torch.no_grad():
                eps_uncond = model(x, t_batch, uncond)
                eps_cond = model(x, t_batch, cond)
            
            # CFG
            eps = eps_uncond + scale * (eps_cond - eps_uncond)
            x = scheduler.step(eps, t, x)
        
        pred = x.permute(0, 2, 1)[0, 0].numpy()
        results[scale] = pred
    
    return results


if __name__ == "__main__":
    ckpt_path = PROJECT_ROOT / "data/output/phase4_checkpoints/best.pt"
    
    print("=" * 60)
    print("FiLM 模型评估")
    print("=" * 60)
    
    print("\n>>> 加载模型...")
    model, obs_norm, act_norm, ds, config = load_model_and_data(ckpt_path)
    print(f"数据集大小: {len(ds):,}")
    
    # 1. Condition 敏感性 (对照文档 10.4)
    print("\n" + "=" * 60)
    print("1. Condition 敏感性测试")
    print("=" * 60)
    diff_cond, diff_input = eval_condition_sensitivity(model, config)
    ratio = diff_cond / diff_input
    print(f"不同 condition 引起的差异: {diff_cond:.4f}")
    print(f"不同 input 引起的差异: {diff_input:.4f}")
    print(f"比值 (cond/input): {ratio:.2f}")
    print(f"[对照] 旧加法版本: 0.15, FiLM 随机初始化: 0.78")
    if ratio > 0.5:
        print(">>> PASS: condition 敏感性良好")
    else:
        print(">>> WARN: condition 敏感性不足")
    
    # 2. cos_sim 评估 (对照文档 9.3)
    print("\n" + "=" * 60)
    print("2. cos_sim 评估 (n=200)")
    print("=" * 60)
    cos_results = eval_cos_sim_with_gt(model, obs_norm, act_norm, ds, config)
    
    print(f"cos_sim(pred, GT):  mean={cos_results['cos_pred_gt'].mean():.3f}, >0.5: {(cos_results['cos_pred_gt']>0.5).mean()*100:.1f}%")
    print(f"cos_sim(pred, nav): mean={cos_results['cos_pred_nav'].mean():.3f}, >0.5: {(cos_results['cos_pred_nav']>0.5).mean()*100:.1f}%")
    print(f"cos_sim(GT, nav):   mean={cos_results['cos_gt_nav'].mean():.3f}, >0.5: {(cos_results['cos_gt_nav']>0.5).mean()*100:.1f}%")
    print(f"[对照] 数据本身 GT-nav: 0.67~0.82")
    
    # 3. 输出多样性
    print("\n" + "=" * 60)
    print("3. 输出多样性 (固定噪声, 不同 condition)")
    print("=" * 60)
    outputs = eval_output_diversity(model, obs_norm, ds, config, n_samples=50)
    print(f"输出 y 分量: mean={outputs[:,0].mean():.3f}, std={outputs[:,0].std():.3f}")
    print(f"输出 x 分量: mean={outputs[:,1].mean():.3f}, std={outputs[:,1].std():.3f}")
    print(f"[对照] 旧模型 std < 0.01, 好的模型 std > 0.3")
    if outputs.std(axis=0).mean() > 0.3:
        print(">>> PASS: 输出多样性良好")
    else:
        print(">>> WARN: 输出多样性不足 (可能仍在输出常数)")
    
    # 4. CFG 效果
    print("\n" + "=" * 60)
    print("4. CFG guidance_scale 效果")
    print("=" * 60)
    cfg_results = eval_cfg_effect(model, obs_norm, ds, config)
    for scale, pred in cfg_results.items():
        print(f"scale={scale}: pred=[{pred[0]:.3f}, {pred[1]:.3f}]")
    
    diff_1_3 = np.linalg.norm(cfg_results[3.0] - cfg_results[1.0])
    diff_1_5 = np.linalg.norm(cfg_results[5.0] - cfg_results[1.0])
    print(f"scale 1->3 变化: {diff_1_3:.3f}")
    print(f"scale 1->5 变化: {diff_1_5:.3f}")
    if diff_1_5 > 0.1:
        print(">>> PASS: CFG 有效")
    else:
        print(">>> WARN: CFG 效果不明显")
    
    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)
