"""
Phase 4 Diffusion Policy 闭环推理

功能：
1. 加载训练好的模型
2. 在地图上放置 Agent
3. 用模型预测动作并执行（MPC 模式）
4. 可视化轨迹
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE4_ROOT = Path(__file__).resolve().parent

# 直接导入模块（不依赖包结构）
import importlib.util

def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入所需模块
config = _import_module("config", PHASE4_ROOT / "config.py")
normalizer_module = _import_module("normalizer", PHASE4_ROOT / "data" / "normalizer.py")
scheduler_module = _import_module("scheduler", PHASE4_ROOT / "diffusion" / "scheduler.py")
unet_module = _import_module("unet1d", PHASE4_ROOT / "model" / "unet1d.py")

ActionNormalizer = normalizer_module.ActionNormalizer
DDPMScheduler = scheduler_module.DDPMScheduler
DDIMScheduler = scheduler_module.DDIMScheduler
UNet1D = unet_module.UNet1D

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiffusionPolicyInference:
    """Diffusion Policy 推理器"""
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        use_ddim: bool = True,
        ddim_steps: int = 20,
    ):
        # 设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # 加载检查点
        self.checkpoint_path = Path(checkpoint_path)
        self._load_model()
        
        # 设置调度器
        if use_ddim:
            self.scheduler = DDIMScheduler(
                num_diffusion_steps=self.config["num_diffusion_steps"],
                num_inference_steps=ddim_steps,
            )
            logger.info(f"Using DDIM scheduler with {ddim_steps} steps")
        else:
            self.scheduler = DDPMScheduler(
                num_diffusion_steps=self.config["num_diffusion_steps"]
            )
            logger.info(f"Using DDPM scheduler with {self.config['num_diffusion_steps']} steps")
    
    def _load_model(self):
        """加载模型和配置"""
        import sys
        import numpy as np
        
        # 修复 numpy 2.x 与 1.x 的兼容性问题
        # numpy 2.x 使用 numpy._core，1.x 使用 numpy.core
        if not hasattr(np, '_core'):
            # 旧版 numpy，创建别名
            sys.modules['numpy._core'] = np.core
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
            sys.modules['numpy._core.numeric'] = np.core.numeric
            sys.modules['numpy._core._multiarray_umath'] = getattr(np.core, '_multiarray_umath', np.core.multiarray)
            logger.info("Applied numpy compatibility patch")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.config = checkpoint["config"]
        
        # 创建模型
        self.model = UNet1D(
            obs_dim=self.config["obs_dim"],
            act_dim=self.config["act_dim"],
            base_channels=self.config["base_channels"],
            cond_dim=self.config["cond_dim"],
            time_dim=self.config["time_dim"],
        ).to(self.device)
        
        # 加载 EMA 权重（更好的生成质量）
        if "ema_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["ema_state_dict"])
            logger.info("Loaded EMA weights")
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model weights")
        
        self.model.eval()
        
        # 加载归一化器
        self.action_normalizer = ActionNormalizer()
        self.action_normalizer.load_state_dict(checkpoint["action_normalizer"])
        
        logger.info(f"Model loaded from {self.checkpoint_path}")
        logger.info(f"Config: history={self.config['history']}, future={self.config['future']}")
    
    @torch.no_grad()
    def predict_action(
        self,
        obs_history: np.ndarray,  # (history, 4): [pos_x, pos_y, vel_x, vel_y]
        num_samples: int = 1,
    ) -> np.ndarray:
        """
        预测未来动作序列
        
        Args:
            obs_history: (history, 4) 历史观测 [位置 + 速度]
            num_samples: 生成样本数
        
        Returns:
            actions: (num_samples, future, 2) 预测的速度序列
        """
        # 准备条件
        obs = torch.from_numpy(obs_history.astype(np.float32))
        obs = obs.reshape(1, -1).to(self.device)  # (1, history * 4)
        obs = obs.expand(num_samples, -1)  # (num_samples, history * 4)
        
        # 采样
        shape = (num_samples, self.config["future"], self.config["act_dim"])
        
        samples = self.scheduler.sample(
            model=self.model,
            shape=shape,
            condition=obs,
            device=self.device,
        )
        
        # 反归一化
        samples = self.action_normalizer.inverse_transform(samples)
        
        return samples.cpu().numpy()
    
    def simulate(
        self,
        start_pos: np.ndarray,
        walkable_mask: np.ndarray,
        max_steps: int = 500,
        num_action_samples: int = 1,
        action_horizon: int = 4,  # MPC 执行多少步（增大以减少推理次数）
    ) -> np.ndarray:
        """
        闭环仿真
        
        Args:
            start_pos: (2,) 起始位置
            walkable_mask: (H, W) 可行走区域
            max_steps: 最大步数
            num_action_samples: 每次预测采样数
            action_horizon: 每次执行多少步预测动作（增大可加速）
        
        Returns:
            trajectory: (T, 2) 轨迹
        """
        history = self.config["history"]
        
        # 初始化状态
        pos = start_pos.copy()
        vel = np.zeros(2)
        
        # 历史记录 (用于构建 obs)
        pos_history = [pos.copy() for _ in range(history)]
        vel_history = [vel.copy() for _ in range(history)]
        
        trajectory = [pos.copy()]
        step = 0
        
        while step < max_steps:
            # 构建 obs
            obs = np.concatenate([
                np.stack(pos_history[-history:], axis=0),  # (history, 2)
                np.stack(vel_history[-history:], axis=0),  # (history, 2)
            ], axis=-1)  # (history, 4)
            
            # 预测动作
            actions = self.predict_action(obs, num_samples=num_action_samples)
            
            # 选择最佳动作（如果采样多个）
            if num_action_samples > 1:
                # 简单策略：选择第一个
                action_seq = actions[0]
            else:
                action_seq = actions[0]
            
            # 执行动作（MPC 模式：执行 action_horizon 步）
            for i in range(min(action_horizon, len(action_seq))):
                if step >= max_steps:
                    break
                    
                vel = action_seq[i]
                new_pos = pos + vel
                
                # 边界检查
                new_pos = np.clip(new_pos, [0, 0], [walkable_mask.shape[0]-1, walkable_mask.shape[1]-1])
                
                # 可行走检查
                ix, iy = int(new_pos[0]), int(new_pos[1])
                if walkable_mask[ix, iy]:
                    pos = new_pos
                # else: 保持原位
                
                # 更新历史
                pos_history.append(pos.copy())
                vel_history.append(vel.copy())
                trajectory.append(pos.copy())
                step += 1
        
        return np.array(trajectory)


def visualize_trajectories(
    trajectories: list[np.ndarray],
    walkable_mask: np.ndarray,
    save_path: Optional[str | Path] = None,
    title: str = "Diffusion Policy Trajectories",
):
    """可视化轨迹"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制可行走区域
    ax.imshow(walkable_mask.T, cmap="gray", origin="lower", alpha=0.5)
    
    # 绘制轨迹
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
    for traj, color in zip(trajectories, colors):
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1, alpha=0.7)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=50, marker="o", zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=50, marker="x", zorder=5)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num_agents", type=int, default=10, help="Number of agents")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per agent")
    parser.add_argument("--use_ddim", action="store_true", default=True, help="Use DDIM")
    parser.add_argument("--ddim_steps", type=int, default=10, help="DDIM steps (fewer = faster)")
    parser.add_argument("--action_horizon", type=int, default=4, help="Steps to execute per prediction")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    
    args = parser.parse_args()
    
    # 默认检查点路径
    if args.checkpoint is None:
        checkpoint_dir = PROJECT_ROOT / "data" / "output" / "phase4_checkpoints"
        best_path = checkpoint_dir / "best.pt"
        latest_path = checkpoint_dir / "latest.pt"
        
        if best_path.exists():
            args.checkpoint = best_path
        elif latest_path.exists():
            args.checkpoint = latest_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    # 加载 walkable mask
    mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Walkable mask not found: {mask_path}")
    
    walkable_mask = np.load(mask_path)
    logger.info(f"Walkable mask shape: {walkable_mask.shape}")
    
    # 创建推理器
    inferencer = DiffusionPolicyInference(
        checkpoint_path=args.checkpoint,
        use_ddim=args.use_ddim,
        ddim_steps=args.ddim_steps,
    )
    
    # 随机选择起始点
    walkable_points = np.argwhere(walkable_mask)
    np.random.seed(42)
    start_indices = np.random.choice(len(walkable_points), args.num_agents, replace=False)
    start_positions = walkable_points[start_indices].astype(np.float32)
    
    # 运行仿真（带进度条）
    from tqdm import tqdm
    logger.info(f"Running simulation for {args.num_agents} agents...")
    trajectories = []
    for i, start_pos in enumerate(tqdm(start_positions, desc="Simulating")):
        traj = inferencer.simulate(
            start_pos=start_pos,
            walkable_mask=walkable_mask,
            max_steps=args.max_steps,
            action_horizon=args.action_horizon,
        )
        trajectories.append(traj)
    
    # 可视化
    output_path = args.output or (PROJECT_ROOT / "data" / "output" / "diffusion_policy_trajectories.png")
    visualize_trajectories(
        trajectories=trajectories,
        walkable_mask=walkable_mask,
        save_path=output_path,
    )
    
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
