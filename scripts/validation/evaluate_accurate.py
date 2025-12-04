"""
Phase 4 模型准确评估脚本
- 正确加载个体导航场
- 正确使用 ObsNormalizer
- 正确计算 cos_sim
"""
import numpy as np
import torch
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/data"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/model"))
sys.path.insert(0, str(PROJECT_ROOT / "src/phase4/diffusion"))

from dataset import TrajectorySlidingWindow
from normalizer import ActionNormalizer, ObsNormalizer
from scheduler import DDIMScheduler
from unet1d import UNet1D

# ============================================================
# 1. 加载检查点
# ============================================================
print("=" * 60)
print("1. 加载检查点")
print("=" * 60)

ckpt_path = PROJECT_ROOT / "data/output/phase4_checkpoints/best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

config = ckpt["config"]
print(f"Config:")
for k, v in config.items():
    print(f"  {k}: {v}")

# ============================================================
# 2. 初始化模型和归一化器
# ============================================================
print("\n" + "=" * 60)
print("2. 初始化模型和归一化器")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = UNet1D(
    obs_dim=config["obs_dim"],
    act_dim=config["act_dim"],
    base_channels=config["base_channels"],
    cond_dim=config["cond_dim"],
    time_dim=config["time_dim"],
)
model.load_state_dict(ckpt["ema_state_dict"])
model.to(device)
model.eval()

action_normalizer = ActionNormalizer(mode="zscore")
action_normalizer.load_state_dict(ckpt["action_normalizer"])
print(f"Action normalizer: mean={action_normalizer.normalizer.mean}, std={action_normalizer.normalizer.std}")

obs_normalizer = ObsNormalizer(mode="zscore", include_nav=True)
obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
print(f"Obs normalizer loaded")

scheduler = DDIMScheduler(
    num_diffusion_steps=config["num_diffusion_steps"],
    num_inference_steps=10,
)

# ============================================================
# 3. 加载数据集（带个体导航场）
# ============================================================
print("\n" + "=" * 60)
print("3. 加载数据集")
print("=" * 60)

nav_fields_dir = PROJECT_ROOT / "data/processed/nav_fields"
nav_global_path = PROJECT_ROOT / "data/processed/nav_baseline.npz"

# 加载全局导航场作为备选
nav_global_data = np.load(nav_global_path)
nav_global = np.stack([nav_global_data['nav_y'], nav_global_data['nav_x']], axis=0)

ds = TrajectorySlidingWindow(
    h5_path=PROJECT_ROOT / "data/output/trajectories.h5",
    history=config["history"],
    future=config["future"],
    nav_field=nav_global,
    nav_fields_dir=nav_fields_dir,
)

print(f"Dataset size: {len(ds):,}")

# ============================================================
# 4. 评估：预测 vs GT vs Nav 方向
# ============================================================
print("\n" + "=" * 60)
print("4. 方向预测评估")
print("=" * 60)

def inference_single(model, scheduler, obs_normed, guidance_scale=1.0):
    """单样本推理"""
    B = 1
    future = config["future"]
    act_dim = config["act_dim"]
    
    # 初始化噪声
    x = torch.randn(B, act_dim, future, device=device)
    global_cond = obs_normed.reshape(B, -1).to(device)
    
    # DDIM 采样（支持 CFG）
    for t in scheduler.timesteps:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        if guidance_scale > 1.0:
            # CFG: 计算无条件和有条件预测
            zeros_cond = torch.zeros_like(global_cond)
            with torch.no_grad():
                eps_uncond = model(x, t_batch, zeros_cond)
                eps_cond = model(x, t_batch, global_cond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            with torch.no_grad():
                eps = model(x, t_batch, global_cond)
        
        x = scheduler.step(eps, t, x)
    
    return x.permute(0, 2, 1)  # (B, future, act_dim)


np.random.seed(42)
num_samples = 100
indices = np.random.choice(len(ds), num_samples, replace=False)

results = {
    "pred_gt_cos": [],
    "pred_nav_cos": [],
    "gt_nav_cos": [],
    "pred_first_step": [],
    "gt_first_step": [],
    "nav_direction": [],
}

guidance_scales = [1.0, 3.0, 5.0]

for gs in guidance_scales:
    print(f"\n--- guidance_scale = {gs} ---")
    
    cos_pred_gt = []
    cos_pred_nav = []
    cos_gt_nav = []
    
    for i, idx in enumerate(indices):
        sample = ds[int(idx)]
        obs = sample["obs"]  # (history, 6)
        action = sample["action"]  # (future, 2) = GT velocity
        
        # 获取 nav 方向（最后一帧）
        nav = obs[-1, 4:6].numpy()  # (2,)
        
        # GT 第一步速度
        gt_vel = action[0].numpy()  # (2,)
        
        # 归一化 obs
        obs_normed = obs_normalizer.transform(obs.unsqueeze(0))  # (1, history, 6)
        
        # 模型预测
        pred_normed = inference_single(model, scheduler, obs_normed, guidance_scale=gs)
        pred = action_normalizer.inverse_transform(pred_normed.cpu())[0]  # (future, 2)
        pred_vel = pred[0].numpy()  # (2,) 第一步预测
        
        # 计算 cos_sim
        def cos_sim(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                return np.nan
            return np.dot(a, b) / (na * nb)
        
        cos_pred_gt.append(cos_sim(pred_vel, gt_vel))
        cos_pred_nav.append(cos_sim(pred_vel, nav))
        cos_gt_nav.append(cos_sim(gt_vel, nav))
    
    cos_pred_gt = np.array(cos_pred_gt)
    cos_pred_nav = np.array(cos_pred_nav)
    cos_gt_nav = np.array(cos_gt_nav)
    
    # 过滤 nan
    valid = ~(np.isnan(cos_pred_gt) | np.isnan(cos_pred_nav) | np.isnan(cos_gt_nav))
    
    print(f"Valid samples: {valid.sum()} / {num_samples}")
    print(f"pred vs GT:  mean={cos_pred_gt[valid].mean():.4f}, >0.5: {(cos_pred_gt[valid]>0.5).mean()*100:.1f}%")
    print(f"pred vs nav: mean={cos_pred_nav[valid].mean():.4f}, >0.5: {(cos_pred_nav[valid]>0.5).mean()*100:.1f}%")
    print(f"GT vs nav:   mean={cos_gt_nav[valid].mean():.4f}, >0.5: {(cos_gt_nav[valid]>0.5).mean()*100:.1f}%")

# ============================================================
# 5. 检查模型输出是否真的依赖 condition
# ============================================================
print("\n" + "=" * 60)
print("5. Condition 敏感性测试")
print("=" * 60)

# 取 10 个不同的样本
test_samples = [ds[int(i)] for i in indices[:10]]

# 固定相同的随机种子
torch.manual_seed(12345)
base_noise = torch.randn(1, config["act_dim"], config["future"], device=device)

preds = []
for sample in test_samples:
    obs = sample["obs"]
    obs_normed = obs_normalizer.transform(obs.unsqueeze(0))
    
    # 使用相同初始噪声
    x = base_noise.clone()
    global_cond = obs_normed.reshape(1, -1).to(device)
    
    for t in scheduler.timesteps:
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, t_batch, global_cond)
        x = scheduler.step(eps, t, x)
    
    pred = x.permute(0, 2, 1)[0, 0].cpu().numpy()  # 第一步预测（归一化空间）
    preds.append(pred)

preds = np.array(preds)
print(f"10 个不同 condition，相同初始噪声，第一步预测（归一化空间）:")
for i, p in enumerate(preds):
    print(f"  sample {i}: [{p[0]:.4f}, {p[1]:.4f}]")

print(f"\n预测的标准差: y={preds[:,0].std():.4f}, x={preds[:,1].std():.4f}")
print(f"预测的变化范围: y=[{preds[:,0].min():.4f}, {preds[:,0].max():.4f}], x=[{preds[:,1].min():.4f}, {preds[:,1].max():.4f}]")

# ============================================================
# 6. 检查 CFG 的效果
# ============================================================
print("\n" + "=" * 60)
print("6. CFG 效果对比")
print("=" * 60)

# 取一个样本
sample = test_samples[0]
obs = sample["obs"]
obs_normed = obs_normalizer.transform(obs.unsqueeze(0))
global_cond = obs_normed.reshape(1, -1).to(device)
zeros_cond = torch.zeros_like(global_cond)

# 固定噪声
torch.manual_seed(999)
x_init = torch.randn(1, config["act_dim"], config["future"], device=device)

def run_with_cfg(x_init, cond, guidance_scale):
    x = x_init.clone()
    for t in scheduler.timesteps:
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_uncond = model(x, t_batch, zeros_cond)
            eps_cond = model(x, t_batch, cond)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x = scheduler.step(eps, t, x)
    return x.permute(0, 2, 1)[0, 0].cpu().numpy()

print("相同样本，不同 guidance_scale:")
for gs in [1.0, 2.0, 3.0, 5.0, 7.0]:
    pred = run_with_cfg(x_init, global_cond, gs)
    print(f"  gs={gs}: [{pred[0]:.4f}, {pred[1]:.4f}]")

# 无条件 vs 有条件
print("\n无条件 vs 有条件 (gs=1.0):")
pred_uncond = run_with_cfg(x_init, zeros_cond, 1.0)
pred_cond = run_with_cfg(x_init, global_cond, 1.0)
print(f"  uncond: [{pred_uncond[0]:.4f}, {pred_uncond[1]:.4f}]")
print(f"  cond:   [{pred_cond[0]:.4f}, {pred_cond[1]:.4f}]")
print(f"  diff:   [{pred_cond[0]-pred_uncond[0]:.4f}, {pred_cond[1]-pred_uncond[1]:.4f}]")

print("\n" + "=" * 60)
print("评估完成")
print("=" * 60)
