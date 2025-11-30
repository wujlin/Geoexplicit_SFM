"""
HDF5 轨迹数据集：
- 读取 trajectories.h5 中的 positions/velocities
- 构造 (obs, action) 对：obs 为过去 history 帧位置，action 为未来 future 帧速度
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryH5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        history: int = 2,
        future: int = 8,
        stride: int = 1,
        agent_ids: Optional[np.ndarray] = None,
    ):
        self.h5_path = Path(h5_path)
        self.history = history
        self.future = future
        self.stride = stride
        self.agent_ids = agent_ids  # 可选：仅使用特定 agent

        with h5py.File(self.h5_path, "r") as f:
            self.pos_shape = f["positions"].shape  # (T, N, 2)
            self.has_vel = "velocities" in f
            if self.has_vel:
                self.vel_shape = f["velocities"].shape
            T, N, _ = self.pos_shape

        if self.agent_ids is None:
            self.agent_ids = np.arange(N, dtype=np.int64)
        else:
            self.agent_ids = np.asarray(agent_ids, dtype=np.int64)

        self.windows_per_agent = (self.pos_shape[0] - (history + future)) // stride
        if self.windows_per_agent <= 0:
            raise ValueError("时间长度不足以构造一个样本")
        self.total = len(self.agent_ids) * self.windows_per_agent
        self._fh = None

    def __len__(self):
        return self.total

    def _ensure_open(self):
        if self._fh is None:
            self._fh = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx):
        self._ensure_open()
        windows_per_agent = self.windows_per_agent
        agent_idx = idx // windows_per_agent
        t_idx = (idx % windows_per_agent) * self.stride
        agent = int(self.agent_ids[agent_idx])

        pos_ds = self._fh["positions"]
        vel_ds = self._fh["velocities"] if "velocities" in self._fh else None

        obs = pos_ds[t_idx : t_idx + self.history, agent, :]  # (history,2)
        if vel_ds is not None:
            action = vel_ds[t_idx + self.history : t_idx + self.history + self.future, agent, :]  # (future,2)
        else:
            # 用位置差分代替速度
            future_pos = pos_ds[t_idx + self.history : t_idx + self.history + self.future + 1, agent, :]
            action = np.diff(future_pos, axis=0)

        obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        action = torch.from_numpy(np.asarray(action, dtype=np.float32))

        return {
            "obs": obs,  # (history,2)
            "action": action,  # (future,2)
            "agent": agent,
            "t0": t_idx,
        }

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
