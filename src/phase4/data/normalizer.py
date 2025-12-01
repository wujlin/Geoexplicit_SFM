"""
数据归一化器：支持 MinMax 和 Z-Score 两种方式
- 训练时 fit 计算统计量
- 推理时 transform/inverse_transform
"""

from __future__ import annotations

import numpy as np
import torch


class Normalizer:
    """数据归一化基类"""
    
    def fit(self, data: np.ndarray) -> "Normalizer":
        raise NotImplementedError
    
    def transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        raise NotImplementedError
    
    def inverse_transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        raise NotImplementedError


class MinMaxNormalizer(Normalizer):
    """MinMax 归一化到 [-1, 1] 范围"""
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.min_val = None
        self.max_val = None
    
    def fit(self, data: np.ndarray) -> "MinMaxNormalizer":
        """计算 min/max，data shape: (..., D)"""
        self.min_val = np.min(data, axis=tuple(range(data.ndim - 1)))
        self.max_val = np.max(data, axis=tuple(range(data.ndim - 1)))
        return self
    
    def transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.min_val is None:
            raise RuntimeError("Normalizer 未 fit")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            min_v = torch.tensor(self.min_val, dtype=data.dtype, device=device)
            max_v = torch.tensor(self.max_val, dtype=data.dtype, device=device)
        else:
            min_v = self.min_val
            max_v = self.max_val
        
        # 归一化到 [0, 1]
        normed = (data - min_v) / (max_v - min_v + self.eps)
        # 转换到 [-1, 1]
        return normed * 2 - 1
    
    def inverse_transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.min_val is None:
            raise RuntimeError("Normalizer 未 fit")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            min_v = torch.tensor(self.min_val, dtype=data.dtype, device=device)
            max_v = torch.tensor(self.max_val, dtype=data.dtype, device=device)
        else:
            min_v = self.min_val
            max_v = self.max_val
        
        # 从 [-1, 1] 转回 [0, 1]
        normed = (data + 1) / 2
        return normed * (max_v - min_v + self.eps) + min_v
    
    def state_dict(self) -> dict:
        return {"min_val": self.min_val, "max_val": self.max_val, "eps": self.eps}
    
    def load_state_dict(self, state: dict) -> "MinMaxNormalizer":
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]
        self.eps = state.get("eps", 1e-8)
        return self


class ZScoreNormalizer(Normalizer):
    """Z-Score 归一化：(x - mean) / std"""
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None
        self.std = None
    
    def fit(self, data: np.ndarray) -> "ZScoreNormalizer":
        """计算 mean/std，data shape: (..., D)"""
        # 沿非最后一维计算
        axes = tuple(range(data.ndim - 1))
        self.mean = np.mean(data, axis=axes)
        self.std = np.std(data, axis=axes)
        return self
    
    def transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.mean is None:
            raise RuntimeError("Normalizer 未 fit")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            mean = torch.tensor(self.mean, dtype=data.dtype, device=device)
            std = torch.tensor(self.std, dtype=data.dtype, device=device)
        else:
            mean = self.mean
            std = self.std
        
        return (data - mean) / (std + self.eps)
    
    def inverse_transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.mean is None:
            raise RuntimeError("Normalizer 未 fit")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            mean = torch.tensor(self.mean, dtype=data.dtype, device=device)
            std = torch.tensor(self.std, dtype=data.dtype, device=device)
        else:
            mean = self.mean
            std = self.std
        
        return data * (std + self.eps) + mean
    
    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}
    
    def load_state_dict(self, state: dict) -> "ZScoreNormalizer":
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state.get("eps", 1e-8)
        return self


class ActionNormalizer:
    """专门用于 action（速度）的归一化器"""
    
    def __init__(self, mode: str = "minmax"):
        if mode == "minmax":
            self.normalizer = MinMaxNormalizer()
        else:
            self.normalizer = ZScoreNormalizer()
        self.mode = mode
    
    def fit(self, actions: np.ndarray) -> "ActionNormalizer":
        """actions shape: (N, T, 2) 或 (N, 2)"""
        flat = actions.reshape(-1, actions.shape[-1])
        self.normalizer.fit(flat)
        return self
    
    def transform(self, actions):
        return self.normalizer.transform(actions)
    
    def inverse_transform(self, actions):
        return self.normalizer.inverse_transform(actions)
    
    def state_dict(self):
        return {"mode": self.mode, "normalizer": self.normalizer.state_dict()}
    
    def load_state_dict(self, state):
        self.mode = state["mode"]
        self.normalizer.load_state_dict(state["normalizer"])
        return self


class ObsNormalizer:
    """
    观测归一化器：分别对 position 和 velocity 归一化
    obs shape: (history, 4) = (history, [pos_x, pos_y, vel_x, vel_y])
    """
    
    def __init__(self, mode: str = "minmax"):
        self.mode = mode
        # 分别归一化 position 和 velocity
        if mode == "minmax":
            self.pos_normalizer = MinMaxNormalizer()
            self.vel_normalizer = MinMaxNormalizer()
        else:
            self.pos_normalizer = ZScoreNormalizer()
            self.vel_normalizer = ZScoreNormalizer()
    
    def fit(self, positions: np.ndarray, velocities: np.ndarray) -> "ObsNormalizer":
        """
        positions shape: (N, 2) 或 (N, T, 2)
        velocities shape: (N, 2) 或 (N, T, 2)
        """
        pos_flat = positions.reshape(-1, 2)
        vel_flat = velocities.reshape(-1, 2)
        self.pos_normalizer.fit(pos_flat)
        self.vel_normalizer.fit(vel_flat)
        return self
    
    def transform(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        obs shape: (..., history, 4) 或 (..., 4)
        前2维是 position，后2维是 velocity
        """
        is_tensor = isinstance(obs, torch.Tensor)
        
        if is_tensor:
            pos = obs[..., :2]
            vel = obs[..., 2:]
            pos_normed = self.pos_normalizer.transform(pos)
            vel_normed = self.vel_normalizer.transform(vel)
            return torch.cat([pos_normed, vel_normed], dim=-1)
        else:
            pos = obs[..., :2]
            vel = obs[..., 2:]
            pos_normed = self.pos_normalizer.transform(pos)
            vel_normed = self.vel_normalizer.transform(vel)
            return np.concatenate([pos_normed, vel_normed], axis=-1)
    
    def inverse_transform(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """反归一化"""
        is_tensor = isinstance(obs, torch.Tensor)
        
        if is_tensor:
            pos = obs[..., :2]
            vel = obs[..., 2:]
            pos_denorm = self.pos_normalizer.inverse_transform(pos)
            vel_denorm = self.vel_normalizer.inverse_transform(vel)
            return torch.cat([pos_denorm, vel_denorm], dim=-1)
        else:
            pos = obs[..., :2]
            vel = obs[..., 2:]
            pos_denorm = self.pos_normalizer.inverse_transform(pos)
            vel_denorm = self.vel_normalizer.inverse_transform(vel)
            return np.concatenate([pos_denorm, vel_denorm], axis=-1)
    
    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "pos_normalizer": self.pos_normalizer.state_dict(),
            "vel_normalizer": self.vel_normalizer.state_dict(),
        }
    
    def load_state_dict(self, state: dict) -> "ObsNormalizer":
        self.mode = state["mode"]
        self.pos_normalizer.load_state_dict(state["pos_normalizer"])
        self.vel_normalizer.load_state_dict(state["vel_normalizer"])
        return self
