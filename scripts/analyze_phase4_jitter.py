"""分析 Phase 4 推理时的抖动来源"""
import numpy as np
import torch
import h5py
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase4.model.unet1d import UNet1D
from phase4.data.normalizer import ObsNormalizer, ActionNormalizer
from phase4.diffusion.scheduler import DDIMScheduler

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

ckpt = torch.load('data/output/phase4_checkpoints/best.pt', map_location=device, weights_only=False)

model = UNet1D(obs_dim=12, act_dim=2, base_channels=128, cond_dim=64, time_dim=64).to(device)
model.load_state_dict(ckpt['ema_state_dict'])
model.eval()

obs_norm = ObsNormalizer()
obs_norm.load_state_dict(ckpt['obs_normalizer'])

act_norm = ActionNormalizer()
act_norm.load_state_dict(ckpt['action_normalizer'])

scheduler = DDIMScheduler(num_diffusion_steps=100, num_inference_steps=20)

# 加载数据
nav_fields_dir = Path('data/processed/nav_fields')
nav_fields = {}
for i in range(35):
    path = nav_fields_dir / f'nav_field_{i:03d}.npz'
    if path.exists():
        data = np.load(path)
        nav_fields[i] = np.stack([data['nav_y'], data['nav_x']], axis=0)

mask = np.load('data/processed/walkable_mask.npy')
H, W = mask.shape

with h5py.File('data/output/trajectories.h5', 'r') as f:
    pos_data = f['positions'][:]
    vel_data = f['velocities'][:]
    dest_data = f['destinations'][:]

# 测试：用 Phase 3 的真实历史，预测单步，比较与 GT 的差异
print("\n=== 测试：单步预测 vs GT ===")

np.random.seed(42)
n_test = 100
guidance_scale = 3.0

pred_angles = []
gt_angles = []
pred_speeds = []
gt_speeds = []
angle_errors = []

for i in range(n_test):
    # 随机选一个样本
    agent = np.random.randint(0, pos_data.shape[1])
    t = np.random.randint(10, 500)
    target = dest_data[t, agent]
    
    # 构建 obs（使用真实历史）
    pos_hist = pos_data[t-2:t, agent, :]  # (2, 2)
    vel_hist = vel_data[t-2:t, agent, :]  # (2, 2)
    
    # 获取 nav
    nav_field = nav_fields[target]
    nav_hist = np.zeros((2, 2), dtype=np.float32)
    for j in range(2):
        y, x = int(np.clip(pos_hist[j, 0], 0, H-1)), int(np.clip(pos_hist[j, 1], 0, W-1))
        nav = nav_field[:, y, x]
        norm = np.linalg.norm(nav)
        nav_hist[j] = nav / norm if norm > 1e-6 else np.zeros(2)
    
    obs = np.concatenate([pos_hist, vel_hist, nav_hist], axis=-1).astype(np.float32)  # (2, 6)
    
    # 预测
    obs_t = torch.tensor(obs).unsqueeze(0).to(device)
    obs_t = obs_norm.transform(obs_t)
    obs_flat = obs_t.reshape(1, -1)
    
    with torch.no_grad():
        samples = scheduler.sample_cfg(model, (1, 8, 2), obs_flat, device, guidance_scale)
        samples = act_norm.inverse_transform(samples)
    
    pred = samples[0, 0].cpu().numpy()  # 第一步预测
    gt = vel_data[t, agent]  # GT
    
    # 计算角度
    pred_angle = np.arctan2(pred[1], pred[0])
    gt_angle = np.arctan2(gt[1], gt[0])
    
    pred_angles.append(pred_angle)
    gt_angles.append(gt_angle)
    pred_speeds.append(np.linalg.norm(pred))
    gt_speeds.append(np.linalg.norm(gt))
    
    # 角度误差
    angle_err = abs(pred_angle - gt_angle)
    angle_err = min(angle_err, 2*np.pi - angle_err)
    angle_errors.append(angle_err)

print(f"样本数: {n_test}")
print(f"平均角度误差: {np.degrees(np.mean(angle_errors)):.1f} deg")
print(f"中位数角度误差: {np.degrees(np.median(angle_errors)):.1f} deg")
print(f"预测速度: {np.mean(pred_speeds):.2f} ± {np.std(pred_speeds):.2f}")
print(f"GT 速度: {np.mean(gt_speeds):.2f} ± {np.std(gt_speeds):.2f}")

# 检查连续预测的方向变化
print("\n=== 测试：连续预测的方向变化 ===")

# 从一个固定位置，连续预测 50 步
agent = 100
t_start = 50
target = dest_data[t_start, agent]

pos = pos_data[t_start, agent].copy()
vel = vel_data[t_start, agent].copy()
nav_field = nav_fields[target]

def get_nav(p):
    y, x = int(np.clip(p[0], 0, H-1)), int(np.clip(p[1], 0, W-1))
    nav = nav_field[:, y, x]
    norm = np.linalg.norm(nav)
    return nav / norm if norm > 1e-6 else np.zeros(2)

pos_hist = [pos.copy(), pos.copy()]
vel_hist = [vel.copy(), vel.copy()]
nav_hist = [get_nav(pos), get_nav(pos)]

pred_velocities = []

for step in range(50):
    obs = np.concatenate([
        np.stack(pos_hist[-2:]),
        np.stack(vel_hist[-2:]),
        np.stack(nav_hist[-2:])
    ], axis=-1).astype(np.float32)
    
    obs_t = torch.tensor(obs).unsqueeze(0).to(device)
    obs_t = obs_norm.transform(obs_t)
    obs_flat = obs_t.reshape(1, -1)
    
    with torch.no_grad():
        samples = scheduler.sample_cfg(model, (1, 8, 2), obs_flat, device, guidance_scale)
        samples = act_norm.inverse_transform(samples)
    
    pred = samples[0, 0].cpu().numpy()
    pred_velocities.append(pred.copy())
    
    # 更新状态
    old_pos = pos.copy()
    new_pos = pos + pred
    new_pos = np.clip(new_pos, [0, 0], [H-1, W-1])
    iy, ix = int(new_pos[0]), int(new_pos[1])
    if mask[iy, ix]:
        pos = new_pos
    
    # 关键修复：vel_hist 记录实际位移，而非预测的 action
    actual_vel = pos - old_pos
    nav = get_nav(pos)
    
    pos_hist.append(pos.copy())
    vel_hist.append(actual_vel)  # 修复：使用实际位移
    nav_hist.append(nav.copy())

# 计算连续预测的方向变化
pred_velocities = np.array(pred_velocities)
speeds = np.linalg.norm(pred_velocities, axis=1)
valid = speeds > 0.1

if valid.sum() >= 2:
    v_valid = pred_velocities[valid]
    angles = np.arctan2(v_valid[:, 1], v_valid[:, 0])
    angle_diff = np.abs(np.diff(angles))
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    
    print(f"连续 50 步预测:")
    print(f"  平均方向变化: {np.degrees(np.mean(angle_diff)):.1f} deg/step")
    print(f"  平均速度: {np.mean(speeds):.2f}")
    print(f"  速度标准差: {np.std(speeds):.2f}")

# 对比：Phase 3 同样 50 步的方向变化
gt_vel = vel_data[t_start:t_start+50, agent, :]
gt_speeds = np.linalg.norm(gt_vel, axis=1)
gt_valid = gt_speeds > 0.1

if gt_valid.sum() >= 2:
    gt_v_valid = gt_vel[gt_valid]
    gt_angles = np.arctan2(gt_v_valid[:, 1], gt_v_valid[:, 0])
    gt_angle_diff = np.abs(np.diff(gt_angles))
    gt_angle_diff = np.minimum(gt_angle_diff, 2*np.pi - gt_angle_diff)
    
    print(f"\nPhase 3 同样 50 步:")
    print(f"  平均方向变化: {np.degrees(np.mean(gt_angle_diff)):.1f} deg/step")
    print(f"  平均速度: {np.mean(gt_speeds):.2f}")
