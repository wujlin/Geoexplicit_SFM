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

# 添加项目根目录/src 到路径，避免包找不到
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from phase4 import config
from phase4.data.dataset import TrajectorySlidingWindow
from phase4.data.normalizer import ActionNormalizer
from phase4.diffusion.scheduler import DDPMScheduler
from phase4.model.unet1d import UNet1D

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
    """Diffusion Policy 训练器"""
    
    def __init__(
        self,
        # 数据配置
        h5_path: str | Path = None,
        history: int = 2,
        future: int = 8,
        # 模型配置
        base_channels: int = 64,
        cond_dim: int = 32,
        time_dim: int = 64,
        # 扩散配置
        num_diffusion_steps: int = 100,
        beta_schedule: str = "linear",
        # 训练配置
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        ema_decay: float = 0.9999,
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
        self.obs_dim = history * 4  # 位置 (2) + 速度 (2) * history
        self.act_dim = 2  # 速度维度
        self.base_channels = base_channels
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        
        # 训练配置
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.ema_decay = ema_decay
        
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
    
    def setup_data(self, val_ratio: float = 0.1):
        """设置数据加载器"""
        logger.info(f"Loading data from {self.h5_path}")
        
        # 创建完整数据集
        full_dataset = TrajectorySlidingWindow(
            h5_path=self.h5_path,
            history=self.history,
            future=self.future,
        )
        
        logger.info(f"Total samples: {len(full_dataset)}")
        
        # 划分训练/验证集
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Train samples: {train_size}, Val samples: {val_size}")
        
        # 计算归一化统计量（从训练集采样）
        logger.info("Computing normalization statistics...")
        self._compute_normalization(train_dataset)
        
        # 创建 DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows 下避免多进程问题
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )
    
    def _compute_normalization(self, dataset, sample_size: int = 10000):
        """从数据集采样计算归一化统计量"""
        sample_size = min(sample_size, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        actions = []
        for idx in indices:
            sample = dataset[idx]
            actions.append(sample["action"].numpy())
        
        actions = np.stack(actions, axis=0)  # (N, future, 2)
        
        self.action_normalizer = ActionNormalizer(mode="minmax")
        self.action_normalizer.fit(actions)
        
        logger.info(f"Action normalizer fitted: min={self.action_normalizer.normalizer.min_val}, "
                   f"max={self.action_normalizer.normalizer.max_val}")
    
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
        """单步训练"""
        self.model.train()
        
        # 获取数据
        obs = batch["obs"].to(self.device)  # (B, history, 4)
        action = batch["action"].to(self.device)  # (B, future, 2)
        
        B = obs.shape[0]
        
        # 归一化 action
        action = self.action_normalizer.transform(action)
        
        # 将 obs 展平作为条件
        global_cond = obs.reshape(B, -1)  # (B, history * 4)
        
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
            
            # 归一化
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
            "config": {
                "history": self.history,
                "future": self.future,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "base_channels": self.base_channels,
                "cond_dim": self.cond_dim,
                "time_dim": self.time_dim,
                "num_diffusion_steps": self.scheduler.num_diffusion_steps,
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
            
            for batch in self.train_loader:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
            
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
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--history", type=int, default=2, help="History steps")
    parser.add_argument("--future", type=int, default=8, help="Future steps to predict")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    trainer = DiffusionPolicyTrainer(
        h5_path=args.h5_path,
        history=args.history,
        future=args.future,
        num_diffusion_steps=args.diffusion_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
