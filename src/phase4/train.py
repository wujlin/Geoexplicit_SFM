"""
Phase 4 Diffusion Policy 训练脚本

功能：
1. 加载 Phase 3 生成的轨迹数据
2. 训练 1D UNet 去噪网络
3. 使用 EMA 平滑模型权重
4. 保存检查点
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

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
dataset_module = _import_module("dataset", PHASE4_ROOT / "data" / "dataset.py")
normalizer_module = _import_module("normalizer", PHASE4_ROOT / "data" / "normalizer.py")
scheduler_module = _import_module("scheduler", PHASE4_ROOT / "diffusion" / "scheduler.py")
unet_module = _import_module("unet1d", PHASE4_ROOT / "model" / "unet1d.py")

TrajectorySlidingWindow = dataset_module.TrajectorySlidingWindow
ActionNormalizer = normalizer_module.ActionNormalizer
ObsNormalizer = normalizer_module.ObsNormalizer
DDPMScheduler = scheduler_module.DDPMScheduler
UNet1D = unet_module.UNet1D

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad = False
    
    def update(self, model: nn.Module):
        with torch.no_grad():
            for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters()):
                shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
    
    def apply(self, model: nn.Module):
        """将 EMA 权重应用到模型"""
        with torch.no_grad():
            for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters()):
                model_p.data.copy_(shadow_p.data)
    
    def state_dict(self):
        return self.shadow.state_dict()
    
    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


class DiffusionPolicyTrainer:
    """Diffusion Policy 训练器
    
    支持 Classifier-Free Guidance (CFG) 训练：
    - cfg_dropout_prob: 训练时随机丢弃 condition 的概率（推荐 0.1）
    - 推理时使用 guidance_scale 增强 condition 影响
    """
    
    def __init__(
        self,
        # 数据配置
        h5_path: str | Path = None,
        history: int = 2,
        future: int = 8,
        # 模型配置 - 增大容量
        base_channels: int = 128,  # 64 -> 128
        cond_dim: int = 64,        # 32 -> 64
        time_dim: int = 64,        # 32 -> 64
        # 扩散配置
        num_diffusion_steps: int = 100,
        beta_schedule: str = "linear",
        # 训练配置
        batch_size: int = 16384,  # 大显存可以用更大 batch
        learning_rate: float = 1e-3,  # 配合超大 batch 提高 lr
        num_epochs: int = 50,     # 30 -> 50，更多训练
        ema_decay: float = 0.9999,
        max_samples_per_epoch: int = 2_000_000,  # 每 epoch 采样 200万
        cfg_dropout_prob: float = 0.1,  # CFG: 训练时丢弃 condition 的概率
        num_workers: int = 8,  # DataLoader workers，根据内存调整
        # 其他
        device: str = "auto",
        checkpoint_dir: str | Path = None,
    ):
        # 设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # 数据路径
        self.h5_path = Path(h5_path or config.TRAJ_H5_PATH)
        self.history = history
        self.future = future
        
        # 模型配置
        # obs = position (2) + velocity (2) + nav_direction (2) = 6 per frame
        self.obs_dim = history * 6  
        self.act_dim = 2  # 速度维度
        self.base_channels = base_channels
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        
        # 训练配置
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.ema_decay = ema_decay
        self.max_samples_per_epoch = max_samples_per_epoch
        self.cfg_dropout_prob = cfg_dropout_prob  # CFG dropout
        self.num_workers = num_workers  # DataLoader workers
        
        # 检查点目录
        self.checkpoint_dir = Path(checkpoint_dir or PROJECT_ROOT / "data" / "output" / "phase4_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.scheduler = DDPMScheduler(
            num_diffusion_steps=num_diffusion_steps,
            beta_schedule=beta_schedule,
        )
        
        self.model = None
        self.ema = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.action_normalizer = None
        self.obs_normalizer = None  # 新增 obs 归一化器
        self.valid_indices_path = None  # 预计算的有效索引路径
        self.nav_field = None  # 全局导航场 (2, H, W)（兼容旧版）
        self.nav_fields_dir = None  # 个体导航场目录
    
    def _load_nav_field(self) -> np.ndarray:
        """加载全局导航场数据（兼容旧版）"""
        nav_path = PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz"
        if not nav_path.exists():
            return None  # 可能使用个体导航场
        
        nav_data = np.load(nav_path)
        nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)
        logger.info(f"Loaded global navigation field: shape={nav_field.shape}")
        return nav_field
    
    def setup_data(self, val_ratio: float = 0.1, valid_indices_path: str | Path = None):
        """设置数据加载器
        
        Args:
            val_ratio: 验证集比例
            valid_indices_path: 预计算的有效索引文件路径（.npy），用于过滤低速样本
        """
        logger.info(f"Loading data from {self.h5_path}")
        
        # 检查是否有个体导航场
        nav_fields_dir = PROJECT_ROOT / "data" / "processed" / "nav_fields"
        if nav_fields_dir.exists():
            self.nav_fields_dir = nav_fields_dir
            logger.info(f"Found individual navigation fields at {nav_fields_dir}")
        else:
            # 回退到全局导航场
            self.nav_field = self._load_nav_field()
        
        # 检查是否存在预计算索引
        if valid_indices_path is None:
            default_path = PROJECT_ROOT / "data" / "output" / "valid_indices.npy"
            if default_path.exists():
                valid_indices_path = default_path
                logger.info(f"Found precomputed valid indices at {valid_indices_path}")
        
        self.valid_indices_path = valid_indices_path
        
        # 创建完整数据集（支持个体导航场）
        full_dataset = TrajectorySlidingWindow(
            h5_path=self.h5_path,
            history=self.history,
            future=self.future,
            valid_indices_path=valid_indices_path,
            nav_field=self.nav_field,  # 全局导航场（可能为 None）
            nav_fields_dir=self.nav_fields_dir,  # 个体导航场目录
        )
        
        total_samples = len(full_dataset)
        if valid_indices_path:
            logger.info(f"Using filtered dataset: {total_samples:,} high-speed samples")
        else:
            logger.info(f"Total samples in dataset: {total_samples:,}")
        
        # 如果数据集太大，进行采样
        if self.max_samples_per_epoch and total_samples > self.max_samples_per_epoch * 1.2:
            # 采样一部分数据用于训练
            sample_size = int(self.max_samples_per_epoch * 1.1)  # 多采一点用于划分验证集
            indices = np.random.choice(total_samples, sample_size, replace=False)
            from torch.utils.data import Subset
            full_dataset = Subset(full_dataset, indices)
            logger.info(f"Sampled {sample_size:,} samples for training")
        
        # 划分训练/验证集
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Train samples: {train_size:,}, Val samples: {val_size:,}")
        
        # 计算归一化统计量（从训练集采样）
        logger.info("Computing normalization statistics...")
        self._compute_normalization(train_dataset)
        
        # 根据平台选择 num_workers
        import platform
        num_workers = self.num_workers if platform.system() != "Windows" else 0
        logger.info(f"DataLoader num_workers={num_workers}")
        
        # 创建 DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    
    def _compute_normalization(self, dataset, sample_size: int = 10000):
        """从数据集采样计算归一化统计量"""
        sample_size = min(sample_size, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        actions = []
        obs_list = []
        for idx in indices:
            sample = dataset[idx]
            actions.append(sample["action"].numpy())
            obs_list.append(sample["obs"].numpy())
        
        actions = np.stack(actions, axis=0)  # (N, future, 2)
        obs_arr = np.stack(obs_list, axis=0)  # (N, history, 6)
        
        # Action 归一化 - 使用 zscore (更适合扩散模型)
        self.action_normalizer = ActionNormalizer(mode="zscore")
        self.action_normalizer.fit(actions)
        
        logger.info(f"Action normalizer fitted (zscore): mean={self.action_normalizer.normalizer.mean}, "
                   f"std={self.action_normalizer.normalizer.std}")
        
        # Obs 归一化 (分别对 position, velocity, nav_direction) - 使用 zscore
        positions = obs_arr[..., :2]  # (N, history, 2)
        velocities = obs_arr[..., 2:4]  # (N, history, 2)
        nav_directions = obs_arr[..., 4:6] if obs_arr.shape[-1] >= 6 else None  # (N, history, 2)
        
        self.obs_normalizer = ObsNormalizer(mode="zscore", include_nav=(nav_directions is not None))
        self.obs_normalizer.fit(positions, velocities, nav_directions)
        
        logger.info(f"Obs normalizer fitted (zscore):")
        logger.info(f"  Position: mean={self.obs_normalizer.pos_normalizer.mean}, "
                   f"std={self.obs_normalizer.pos_normalizer.std}")
        logger.info(f"  Velocity: mean={self.obs_normalizer.vel_normalizer.mean}, "
                   f"std={self.obs_normalizer.vel_normalizer.std}")
        if self.obs_normalizer.include_nav:
            nav_norm = self.obs_normalizer.nav_normalizer
            # IdentityNormalizer 没有 mean/std，显示 scale
            if hasattr(nav_norm, 'mean'):
                logger.info(f"  NavDir: mean={nav_norm.mean}, std={nav_norm.std}")
            elif hasattr(nav_norm, 'scale'):
                logger.info(f"  NavDir: IdentityNormalizer(scale={nav_norm.scale})")
    
    def setup_model(self):
        """初始化模型和优化器"""
        self.model = UNet1D(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            base_channels=self.base_channels,
            cond_dim=self.cond_dim,
            time_dim=self.time_dim,
        ).to(self.device)
        
        # 统计参数量
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        
        # EMA
        self.ema = EMA(self.model, decay=self.ema_decay)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
    
    def train_step(self, batch: dict) -> float:
        """单步训练 (支持 CFG)"""
        self.model.train()
        
        # 获取数据
        obs = batch["obs"].to(self.device)  # (B, history, 4)
        action = batch["action"].to(self.device)  # (B, future, 2)
        
        B = obs.shape[0]
        
        # 归一化 obs 和 action
        obs = self.obs_normalizer.transform(obs)
        action = self.action_normalizer.transform(action)
        
        # 将 obs 展平作为条件
        global_cond = obs.reshape(B, -1)  # (B, history * 6)
        
        # CFG: 随机丢弃 condition
        if self.cfg_dropout_prob > 0:
            # 为每个样本独立决定是否丢弃 condition
            dropout_mask = torch.rand(B, device=self.device) < self.cfg_dropout_prob
            # 将被选中的样本的 condition 置零
            global_cond = global_cond * (~dropout_mask).unsqueeze(-1).float()
        
        # 采样噪声和时间步
        noise = torch.randn_like(action)
        timesteps = self.scheduler.sample_timesteps(B, self.device)
        
        # 前向扩散：添加噪声
        noisy_action = self.scheduler.add_noise(action, noise, timesteps)
        
        # 模型预测噪声
        # UNet 期望输入 (B, C, T)，所以需要转置
        noisy_action_input = noisy_action.permute(0, 2, 1)  # (B, future, 2) -> (B, 2, future)
        predicted_noise = self.model(noisy_action_input, timesteps, global_cond)
        predicted_noise = predicted_noise.permute(0, 2, 1)  # (B, 2, future) -> (B, future, 2)
        
        # 计算 loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新 EMA
        self.ema.update(self.model)
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            obs = batch["obs"].to(self.device)
            action = batch["action"].to(self.device)
            
            B = obs.shape[0]
            
            # 归一化 obs 和 action
            obs = self.obs_normalizer.transform(obs)
            action = self.action_normalizer.transform(action)
            global_cond = obs.reshape(B, -1)
            
            # 采样噪声
            noise = torch.randn_like(action)
            timesteps = self.scheduler.sample_timesteps(B, self.device)
            
            noisy_action = self.scheduler.add_noise(action, noise, timesteps)
            
            noisy_action_input = noisy_action.permute(0, 2, 1)
            predicted_noise = self.model(noisy_action_input, timesteps, global_cond)
            predicted_noise = predicted_noise.permute(0, 2, 1)
            
            loss = F.mse_loss(predicted_noise, noise)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "action_normalizer": self.action_normalizer.state_dict(),
            "obs_normalizer": self.obs_normalizer.state_dict(),  # 新增
            "config": {
                "history": self.history,
                "future": self.future,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "base_channels": self.base_channels,
                "cond_dim": self.cond_dim,
                "time_dim": self.time_dim,
                "num_diffusion_steps": self.scheduler.num_diffusion_steps,
                "cfg_dropout_prob": self.cfg_dropout_prob,  # 保存 CFG 配置
            }
        }
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss {loss:.6f}")
        
        # 每 10 个 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, path: str | Path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.action_normalizer.load_state_dict(checkpoint["action_normalizer"])
        
        return checkpoint["epoch"], checkpoint["loss"]
    
    def train(self):
        """完整训练流程"""
        logger.info("="*60)
        logger.info("Starting Diffusion Policy Training")
        logger.info("="*60)
        
        # 设置数据和模型
        self.setup_data()
        self.setup_model()
        
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            # 使用 tqdm 显示进度条
            pbar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                leave=False,
                ncols=100
            )
            
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # 更新进度条显示
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg': f'{epoch_loss/num_batches:.4f}'
                })
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # 验证
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            # 检查是否是最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss, is_best=is_best)
            
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )
        
        # 保存训练曲线
        np.savez(
            self.checkpoint_dir / "training_history.npz",
            train_losses=train_losses,
            val_losses=val_losses,
        )
        
        logger.info("="*60)
        logger.info(f"Training completed. Best val loss: {best_val_loss:.6f}")
        logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy")
    parser.add_argument("--h5_path", type=str, default=None, help="Path to trajectories.h5")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size (larger = faster)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--history", type=int, default=2, help="History steps")
    parser.add_argument("--future", type=int, default=8, help="Future steps to predict")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--max_samples", type=int, default=2_000_000, help="Max samples per epoch")
    parser.add_argument("--base_channels", type=int, default=128, help="UNet base channels")
    parser.add_argument("--cond_dim", type=int, default=64, help="Condition embedding dim")
    parser.add_argument("--cfg_dropout", type=float, default=0.1, help="CFG condition dropout probability")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers (increase if GPU idle)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--valid_indices", type=str, default=None, 
                        help="Path to precomputed valid indices (.npy). If not specified, will look for data/output/valid_indices.npy")
    
    args = parser.parse_args()
    
    trainer = DiffusionPolicyTrainer(
        h5_path=args.h5_path,
        history=args.history,
        future=args.future,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        num_diffusion_steps=args.diffusion_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_samples_per_epoch=args.max_samples,
        cfg_dropout_prob=args.cfg_dropout,
        num_workers=args.num_workers,
        device=args.device,
    )
    
    # 设置数据时传入 valid_indices 路径
    valid_indices_path = args.valid_indices if args.valid_indices else None
    trainer.setup_data(valid_indices_path=valid_indices_path)
    trainer.setup_model()
    trainer.train()


if __name__ == "__main__":
    main()
