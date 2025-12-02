"""
Phase 4 Nav-Guided Inference

在推理时对模型输出进行 Nav 方向的 guidance，
弥补当前模型条件响应不足的问题。

用法:
    python scripts/nav_guided_inference.py
    python scripts/nav_guided_inference.py --guidance 3.0 --num_agents 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import importlib.util

def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

PHASE4_ROOT = PROJECT_ROOT / "src" / "phase4"
normalizer_module = _import_module("normalizer", PHASE4_ROOT / "data" / "normalizer.py")
scheduler_module = _import_module("scheduler", PHASE4_ROOT / "diffusion" / "scheduler.py")
unet_module = _import_module("unet1d", PHASE4_ROOT / "model" / "unet1d.py")

ActionNormalizer = normalizer_module.ActionNormalizer
ObsNormalizer = normalizer_module.ObsNormalizer
DDIMScheduler = scheduler_module.DDIMScheduler
UNet1D = unet_module.UNet1D


class NavGuidedDiffusionPolicy:
    """带 Nav Guidance 的 Diffusion Policy"""
    
    def __init__(self, checkpoint_path: Path, guidance_scale: float = 2.0, device: str = "cpu"):
        self.device = device
        self.guidance_scale = guidance_scale
        
        # numpy 兼容性
        if not hasattr(np, '_core'):
            sys.modules['numpy._core'] = np.core
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = checkpoint["config"]
        
        self.model = UNet1D(
            obs_dim=self.config["obs_dim"],
            act_dim=self.config["act_dim"],
            base_channels=self.config["base_channels"],
            cond_dim=self.config["cond_dim"],
            time_dim=self.config["time_dim"],
        ).to(device)
        
        if "ema_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["ema_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.action_normalizer = ActionNormalizer()
        self.action_normalizer.load_state_dict(checkpoint["action_normalizer"])
        
        self.obs_normalizer = ObsNormalizer()
        self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        
        self.scheduler = DDIMScheduler(
            num_diffusion_steps=self.config["num_diffusion_steps"],
            num_inference_steps=10,  # 快速推理
        )
        
        self.history = self.config["history"]
        self.future = self.config["future"]
    
    @torch.no_grad()
    def predict(self, obs: np.ndarray, nav_direction: np.ndarray) -> np.ndarray:
        """
        预测带 nav guidance 的动作序列
        
        Args:
            obs: (history, 6) = [pos, vel, nav]
            nav_direction: (2,) 单位向量，表示目标方向
            
        Returns:
            action: (future, 2) velocity 序列
        """
        B = 1
        
        # 准备输入
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_norm = self.obs_normalizer.transform(obs_tensor)
        cond = obs_norm.reshape(B, -1)
        
        # Diffusion 采样
        shape = (B, self.future, 2)
        pred_norm = self.scheduler.sample(self.model, shape, cond, device=self.device)
        
        # 反归一化
        pred = self.action_normalizer.inverse_transform(pred_norm).cpu().numpy()[0]
        
        # Nav Guidance: 将预测方向推向 nav
        nav = np.array(nav_direction, dtype=np.float32)
        nav = nav / (np.linalg.norm(nav) + 1e-8)
        
        guided_pred = []
        g = self.guidance_scale * 0.3  # guidance 强度
        
        for t in range(self.future):
            vel = pred[t]
            speed = np.linalg.norm(vel)
            
            if speed < 0.01:
                # 静止时直接用 nav 方向
                guided_pred.append(nav * 0.8)
            else:
                vel_dir = vel / speed
                # 混合方向
                mixed_dir = vel_dir * (1 - g) + nav * g
                mixed_dir = mixed_dir / (np.linalg.norm(mixed_dir) + 1e-8)
                guided_pred.append(mixed_dir * speed)
        
        return np.array(guided_pred)


def run_simulation(
    policy: NavGuidedDiffusionPolicy,
    nav_field: np.ndarray,
    walkable_mask: np.ndarray,
    distance_field: np.ndarray,
    start_positions: np.ndarray,
    max_steps: int = 300,
):
    """运行闭环仿真"""
    H, W = walkable_mask.shape
    num_agents = len(start_positions)
    history = policy.history
    
    # 初始化
    trajectories = []
    initial_distances = []
    final_distances = []
    
    for agent_idx in tqdm(range(num_agents), desc="Simulating"):
        pos = start_positions[agent_idx].copy().astype(np.float32)
        
        # 获取初始 nav
        y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
        nav = nav_field[:, y, x]
        nav_norm = np.linalg.norm(nav)
        if nav_norm > 1e-6:
            nav = nav / nav_norm
        else:
            nav = np.array([0.0, 0.0])
        
        vel = nav * 0.8  # 初始速度
        
        # 历史记录
        pos_history = [pos.copy() for _ in range(history)]
        vel_history = [vel.copy() for _ in range(history)]
        nav_history = [nav.copy() for _ in range(history)]
        
        trajectory = [pos.copy()]
        initial_distances.append(distance_field[y, x])
        
        action_horizon = 4  # MPC 执行步数
        
        for step in range(max_steps):
            # 构建 obs
            obs = np.concatenate([
                np.stack(pos_history[-history:], axis=0),
                np.stack(vel_history[-history:], axis=0),
                np.stack(nav_history[-history:], axis=0),
            ], axis=-1)  # (history, 6)
            
            # 获取当前 nav
            y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
            current_nav = nav_field[:, y, x]
            nav_norm = np.linalg.norm(current_nav)
            if nav_norm > 1e-6:
                current_nav = current_nav / nav_norm
            else:
                current_nav = np.array([0.0, 0.0])
            
            # 预测
            action_seq = policy.predict(obs, current_nav)
            
            # 执行动作
            for i in range(min(action_horizon, len(action_seq))):
                vel = action_seq[i]
                new_pos = pos + vel
                new_pos = np.clip(new_pos, [0, 0], [H-1, W-1])
                
                # 可行走检查
                iy, ix = int(new_pos[0]), int(new_pos[1])
                if walkable_mask[iy, ix]:
                    pos = new_pos
                
                # 更新 nav
                y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
                nav = nav_field[:, y, x]
                nav_norm = np.linalg.norm(nav)
                if nav_norm > 1e-6:
                    nav = nav / nav_norm
                
                pos_history.append(pos.copy())
                vel_history.append(vel.copy())
                nav_history.append(nav.copy())
                trajectory.append(pos.copy())
            
            # 检查是否到达 sink
            y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
            if distance_field[y, x] < 5:
                break
        
        trajectories.append(np.array(trajectory))
        y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
        final_distances.append(distance_field[y, x])
    
    return trajectories, np.array(initial_distances), np.array(final_distances)


def main():
    parser = argparse.ArgumentParser(description="Nav-Guided Diffusion Policy Inference")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=2.0, help="Nav guidance scale")
    parser.add_argument("--num_agents", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    # 路径
    if args.checkpoint is None:
        args.checkpoint = PROJECT_ROOT / "data" / "output" / "phase4_checkpoints" / "best.pt"
    else:
        args.checkpoint = Path(args.checkpoint)
    
    if args.output is None:
        args.output = PROJECT_ROOT / "data" / "output" / "nav_guided_results"
    else:
        args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Nav-Guided Diffusion Policy Inference")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Guidance scale: {args.guidance}")
    print(f"Num agents: {args.num_agents}")
    print()
    
    # 加载数据
    print("Loading data...")
    nav_data = np.load(PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz")
    nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)
    
    walkable_mask = np.load(PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy")
    distance_field = np.load(PROJECT_ROOT / "data" / "processed" / "distance_field.npy")
    
    H, W = walkable_mask.shape
    print(f"Map size: {H} x {W}")
    
    # 选择起点
    np.random.seed(42)
    walkable_points = np.argwhere(walkable_mask)
    nav_mag = np.sqrt(nav_field[0]**2 + nav_field[1]**2)
    point_mags = nav_mag[walkable_points[:, 0], walkable_points[:, 1]]
    good_indices = np.where(point_mags > 0.5)[0]
    
    if len(good_indices) >= args.num_agents:
        selected = np.random.choice(good_indices, args.num_agents, replace=False)
    else:
        selected = np.random.choice(len(walkable_points), args.num_agents, replace=False)
    
    start_positions = walkable_points[selected].astype(np.float32)
    print(f"Selected {len(start_positions)} start positions")
    
    # 创建 Policy
    print("Loading model...")
    policy = NavGuidedDiffusionPolicy(args.checkpoint, guidance_scale=args.guidance)
    print(f"Model loaded (future={policy.future}, history={policy.history})")
    print()
    
    # 运行仿真
    print("Running simulation...")
    trajectories, init_dist, final_dist = run_simulation(
        policy, nav_field, walkable_mask, distance_field,
        start_positions, max_steps=args.max_steps
    )
    
    # 计算指标
    dist_change = final_dist - init_dist
    approaching_rate = (dist_change < 0).mean()
    
    print()
    print("="*60)
    print("Results")
    print("="*60)
    print(f"Approaching rate: {approaching_rate*100:.1f}%")
    print(f"Mean distance change: {dist_change.mean():.1f} px")
    print(f"Initial distance: {init_dist.mean():.1f} +/- {init_dist.std():.1f}")
    print(f"Final distance: {final_dist.mean():.1f} +/- {final_dist.std():.1f}")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(walkable_mask.T, cmap="gray", origin="lower", alpha=0.5)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
    for traj, color in zip(trajectories, colors):
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1, alpha=0.7)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=30, marker="o", zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=30, marker="x", zorder=5)
    
    ax.set_title(f"Nav-Guided Diffusion Trajectories (n={len(trajectories)}, guidance={args.guidance})\n"
                 f"Approaching rate: {approaching_rate*100:.1f}%")
    ax.set_aspect("equal")
    
    fig_path = args.output / "trajectories.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
