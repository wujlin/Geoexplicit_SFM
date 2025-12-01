"""
HDF5 轨迹数据集：滑动窗口读取 (observation, action)
- observation: 过去 history 步的位置、速度、导航方向 (shape: history, 6)
- action: 未来 future 步的速度序列 (shape: future, 2)
- 支持加载预计算的有效样本索引（过滤低速样本）

使用方法：
1. 先运行预处理：python scripts/precompute_valid_indices.py
2. 训练时指定索引文件：TrajectorySlidingWindow(..., valid_indices_path="data/output/valid_indices.npy")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectorySlidingWindow(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        history: int = 2,
        future: int = 8,
        stride: int = 1,
        agent_ids: Optional[np.ndarray] = None,
        valid_indices_path: Optional[str | Path] = None,  # 预计算的有效索引文件
        nav_field: Optional[np.ndarray] = None,  # (2, H, W) 导航场方向
    ):
        self.h5_path = Path(h5_path)
        self.history = history
        self.future = future
        self.stride = stride
        self.nav_field = nav_field  # 导航场条件

        with h5py.File(self.h5_path, "r") as f:
            self.pos_shape = f["positions"].shape  # (T, N, 2)
            self.has_vel = "velocities" in f
            T, N, _ = self.pos_shape
        
        if agent_ids is None:
            self.agent_ids = np.arange(N, dtype=np.int64)
        else:
            self.agent_ids = np.asarray(agent_ids, dtype=np.int64)
        
        # 加载预计算的有效索引
        if valid_indices_path is not None:
            valid_indices_path = Path(valid_indices_path)
            if valid_indices_path.exists():
                self.valid_indices = np.load(valid_indices_path)
                self.total = len(self.valid_indices)
                print(f"Loaded {self.total:,} valid indices from {valid_indices_path}")
            else:
                raise FileNotFoundError(
                    f"Valid indices file not found: {valid_indices_path}\n"
                    f"Run: python scripts/precompute_valid_indices.py"
                )
        else:
            # 不过滤，使用所有样本
            windows_per_agent = (T - (self.history + self.future)) // self.stride
            if windows_per_agent <= 0:
                raise ValueError("时间长度不足以构造样本")
            
            # 构建所有索引
            indices_list = []
            for agent in self.agent_ids:
                for w in range(windows_per_agent):
                    t_idx = w * self.stride
                    indices_list.append((int(agent), t_idx))
            self.valid_indices = np.array(indices_list, dtype=np.int64)
            self.total = len(self.valid_indices)
            print(f"Using all {self.total:,} samples (no filtering)")
        
        self._fh = None

    def __len__(self):
        return self.total

    def _ensure_open(self):
        if self._fh is None:
            self._fh = h5py.File(self.h5_path, "r")
    
    def _get_nav_direction(self, pos: np.ndarray) -> np.ndarray:
        """获取位置处的导航方向 (2,)"""
        if self.nav_field is None:
            return np.zeros(2, dtype=np.float32)
        
        H, W = self.nav_field.shape[1], self.nav_field.shape[2]
        y = int(np.clip(pos[0], 0, H - 1))
        x = int(np.clip(pos[1], 0, W - 1))
        nav_dir = self.nav_field[:, y, x]
        return nav_dir.astype(np.float32)

    def __getitem__(self, idx):
        self._ensure_open()
        
        agent, t_idx = self.valid_indices[idx]
        agent = int(agent)
        t_idx = int(t_idx)

        pos_ds = self._fh["positions"]
        vel_ds = self._fh["velocities"] if self.has_vel else None

        # 历史 obs
        pos_hist = pos_ds[t_idx : t_idx + self.history, agent, :]  # (history,2)
        if vel_ds is not None:
            vel_hist = vel_ds[t_idx : t_idx + self.history, agent, :]  # (history,2)
        else:
            pos_hist_ext = pos_ds[t_idx : t_idx + self.history + 1, agent, :]
            vel_hist = np.diff(pos_hist_ext, axis=0)
        
        # 获取每个历史帧的导航方向
        nav_hist = np.zeros((self.history, 2), dtype=np.float32)
        for i in range(self.history):
            nav_hist[i] = self._get_nav_direction(pos_hist[i])

        # obs: position + velocity + nav_direction
        obs = np.concatenate([pos_hist, vel_hist, nav_hist], axis=-1)  # (history, 6)

        # 未来 action（速度）
        if vel_ds is not None:
            action = vel_ds[t_idx + self.history : t_idx + self.history + self.future, agent, :]  # (future,2)
        else:
            pos_future_ext = pos_ds[t_idx + self.history : t_idx + self.history + self.future + 1, agent, :]
            action = np.diff(pos_future_ext, axis=0)

        obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        action = torch.from_numpy(np.asarray(action, dtype=np.float32))

        return {
            "obs": obs,  # (history, 6) = [pos(2) + vel(2) + nav(2)]
            "action": action,  # (future, 2)
            "agent": agent,
            "t0": t_idx,
        }

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
