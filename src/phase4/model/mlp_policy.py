"""
简单 MLP Policy：直接学习 nav → velocity 的映射

这是一个轻量级替代方案，适用于近似确定性的策略学习任务。
Diffusion Policy 适合多模态分布，但我们的任务是确定性的。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """
    简单 MLP 策略网络
    
    输入: obs = [pos(2), vel(2), nav(2)] * history = 12 维
    输出: action = velocity(2) * future = 16 维
    
    但实际上，最重要的是 nav 和 vel 的信息
    """
    
    def __init__(self, obs_dim: int = 12, act_dim: int = 2, future: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.future = future
        
        # 简单的 MLP
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim * future),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim) 或 (B, history, 6)
        返回: (B, future, act_dim)
        """
        if obs.dim() == 3:
            obs = obs.reshape(obs.shape[0], -1)
        
        out = self.net(obs)  # (B, act_dim * future)
        return out.reshape(-1, self.future, self.act_dim)


class NavConditionedPolicy(nn.Module):
    """
    更简单的策略：只用 nav 和 vel 作为输入
    
    核心洞察：position 对于预测下一步 velocity 几乎没有用
    真正重要的是：当前速度 + 目标方向
    """
    
    def __init__(self, hidden_dim: int = 128, future: int = 8):
        super().__init__()
        self.future = future
        
        # 输入: vel(2) + nav(2) = 4 维
        # 输出: future_vel(2) * future = 16 维
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * future),
        )
    
    def forward(self, vel: torch.Tensor, nav: torch.Tensor) -> torch.Tensor:
        """
        vel: (B, 2) 当前速度
        nav: (B, 2) 目标方向
        返回: (B, future, 2) 未来速度序列
        """
        x = torch.cat([vel, nav], dim=-1)  # (B, 4)
        out = self.net(x)  # (B, future * 2)
        return out.reshape(-1, self.future, 2)


def train_mlp_policy(
    h5_path: str,
    nav_path: str,
    epochs: int = 20,
    batch_size: int = 8192,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = None,
):
    """快速训练 MLP Policy"""
    import h5py
    import numpy as np
    from pathlib import Path
    
    print("加载数据...")
    h5 = h5py.File(h5_path, "r")
    positions = h5["positions"][:]  # (T, N, 2)
    velocities = h5["velocities"][:]
    h5.close()
    
    nav_data = np.load(nav_path)
    nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)  # (2, H, W)
    
    T, N, _ = positions.shape
    H, W = nav_field.shape[1], nav_field.shape[2]
    
    history = 2
    future = 8
    
    print("构建训练数据...")
    # 采样训练数据
    num_samples = min(500000, (T - history - future) * N)
    
    np.random.seed(42)
    sample_t = np.random.randint(0, T - history - future, num_samples)
    sample_n = np.random.randint(0, N, num_samples)
    
    # 提取数据
    obs_list = []
    action_list = []
    
    for i in range(num_samples):
        t, n = sample_t[i], sample_n[i]
        
        # obs: [pos, vel, nav] * history
        obs = np.zeros((history, 6), dtype=np.float32)
        for h in range(history):
            pos = positions[t + h, n]
            vel = velocities[t + h, n]
            y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
            nav = nav_field[:, y, x]
            obs[h] = np.concatenate([pos, vel, nav])
        
        # action: future velocities
        action = velocities[t + history : t + history + future, n]
        
        # 过滤静止样本
        if np.linalg.norm(action.mean(axis=0)) > 0.1:
            obs_list.append(obs.flatten())
            action_list.append(action)
    
    obs_array = np.array(obs_list, dtype=np.float32)
    action_array = np.array(action_list, dtype=np.float32)
    
    print(f"有效样本数: {len(obs_array)}")
    
    # 归一化
    obs_mean = obs_array.mean(axis=0)
    obs_std = obs_array.std(axis=0) + 1e-8
    action_mean = action_array.mean(axis=(0, 1))
    action_std = action_array.std(axis=(0, 1)) + 1e-8
    
    obs_array = (obs_array - obs_mean) / obs_std
    action_array = (action_array - action_mean) / action_std
    
    # 转换为 tensor
    obs_tensor = torch.tensor(obs_array, device=device)
    action_tensor = torch.tensor(action_array, device=device)
    
    # 创建模型
    model = MLPPolicy(obs_dim=12, act_dim=2, future=8, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 训练
    dataset_size = len(obs_tensor)
    best_loss = float("inf")
    
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(dataset_size, device=device)
        obs_tensor = obs_tensor[perm]
        action_tensor = action_tensor[perm]
        
        epoch_loss = 0.0
        num_batches = 0
        
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            obs_batch = obs_tensor[start:end]
            action_batch = action_tensor[start:end]
            
            pred = model(obs_batch)
            loss = F.mse_loss(pred, action_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "config": {"obs_dim": 12, "act_dim": 2, "future": 8, "hidden_dim": 256},
                }, save_path)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")
    
    print()
    print(f"训练完成! Best loss: {best_loss:.6f}")
    if save_path:
        print(f"模型保存至: {save_path}")
    
    return model, obs_mean, obs_std, action_mean, action_std


if __name__ == "__main__":
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    
    train_mlp_policy(
        h5_path=str(PROJECT_ROOT / "data" / "output" / "trajectories.h5"),
        nav_path=str(PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz"),
        epochs=20,
        batch_size=8192,
        save_path=str(PROJECT_ROOT / "data" / "output" / "phase4_checkpoints" / "mlp_policy.pt"),
    )
