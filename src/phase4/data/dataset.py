"""
HDF5 轨迹数据集：滑动窗口读取 (observation, action)
- observation: 过去 history 步的位置与速度 (shape: history, 4)
- action: 未来 future 步的速度序列 (shape: future, 2)
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
    ):
        self.h5_path = Path(h5_path)
        self.history = history
        self.future = future
        self.stride = stride
        self.agent_ids = agent_ids

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

        self.windows_per_agent = (self.pos_shape[0] - (self.history + self.future)) // self.stride
        if self.windows_per_agent <= 0:
            raise ValueError("时间长度不足以构造样本")
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
        vel_ds = self._fh["velocities"] if self.has_vel else None

        # 历史 obs
        pos_hist = pos_ds[t_idx : t_idx + self.history, agent, :]  # (history,2)
        if vel_ds is not None:
            vel_hist = vel_ds[t_idx : t_idx + self.history, agent, :]  # (history,2)
        else:
            pos_hist_ext = pos_ds[t_idx : t_idx + self.history + 1, agent, :]
            vel_hist = np.diff(pos_hist_ext, axis=0)

        obs = np.concatenate([pos_hist, vel_hist], axis=-1)  # (history,4)

        # 未来 action（速度）
        if vel_ds is not None:
            action = vel_ds[t_idx + self.history : t_idx + self.history + self.future, agent, :]  # (future,2)
        else:
            pos_future_ext = pos_ds[t_idx + self.history : t_idx + self.history + self.future + 1, agent, :]
            action = np.diff(pos_future_ext, axis=0)

        obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        action = torch.from_numpy(np.asarray(action, dtype=np.float32))

        return {
            "obs": obs,  # (history,4)
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
