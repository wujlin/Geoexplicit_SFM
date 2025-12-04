"""
HDF5 轨迹数据集：滑动窗口读取 (observation, action)
- observation: 过去 history 步的位置、速度、导航方向 (shape: history, 6)
- action: 未来 future 步的速度序列 (shape: future, 2)
- 支持加载预计算的有效样本索引（过滤低速样本）
- 支持个体目的地：每个 agent 有独立的目的地，使用对应的导航场

使用方法：
1. 先运行预处理：python scripts/precompute_valid_indices.py
2. 训练时指定索引文件：TrajectorySlidingWindow(..., valid_indices_path="data/output/valid_indices.npy")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import json

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
        nav_field: Optional[np.ndarray] = None,  # (2, H, W) 全局导航场（兼容旧版）
        nav_fields_dir: Optional[str | Path] = None,  # 个体导航场目录
    ):
        self.h5_path = Path(h5_path)
        self.history = history
        self.future = future
        self.stride = stride
        self.nav_field = nav_field  # 全局导航场（兼容旧版）
        
        # 加载个体导航场
        self.nav_fields: Dict[int, np.ndarray] = {}
        self.has_individual_nav = False
        if nav_fields_dir is not None:
            self._load_nav_fields(Path(nav_fields_dir))

        with h5py.File(self.h5_path, "r") as f:
            self.pos_shape = f["positions"].shape  # (T, N, 2)
            self.has_vel = "velocities" in f
            self.has_dest = "destinations" in f
            T, N, _ = self.pos_shape
            
            # 加载目的地信息（如果存在）
            if self.has_dest:
                self.destinations = f["destinations"][:]  # (T, N)
                print(f"Loaded destinations: shape={self.destinations.shape}")
            else:
                self.destinations = None
        
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
    
    def _load_nav_fields(self, nav_fields_dir: Path):
        """加载所有个体导航场到内存"""
        index_path = nav_fields_dir / "nav_fields_index.json"
        if not index_path.exists():
            print(f"Warning: nav_fields_index.json not found in {nav_fields_dir}")
            return
        
        with open(index_path) as f:
            index = json.load(f)
        
        print(f"Loading {index['num_sinks']} individual navigation fields...")
        
        for sink_id in index["sink_ids"]:
            path = nav_fields_dir / f"nav_field_{sink_id:03d}.npz"
            data = np.load(path)
            # 存储为 (2, H, W) 格式
            self.nav_fields[sink_id] = np.stack([data["nav_y"], data["nav_x"]], axis=0)
        
        self.has_individual_nav = True
        print(f"  Loaded {len(self.nav_fields)} navigation fields")

    def __len__(self):
        return self.total

    def _ensure_open(self):
        if self._fh is None:
            self._fh = h5py.File(self.h5_path, "r")
    
    def _get_nav_direction(self, pos: np.ndarray, dest: Optional[int] = None) -> np.ndarray:
        """
        获取位置处的导航方向 (2,)
        
        Args:
            pos: (2,) 位置 [y, x]
            dest: 目的地 sink ID（如果有个体导航场）
        """
        # 优先使用个体导航场
        if self.has_individual_nav and dest is not None and dest in self.nav_fields:
            nav = self.nav_fields[dest]
            H, W = nav.shape[1], nav.shape[2]
            y = int(np.clip(pos[0], 0, H - 1))
            x = int(np.clip(pos[1], 0, W - 1))
            return nav[:, y, x].astype(np.float32)
        
        # 回退到全局导航场
        if self.nav_field is not None:
            H, W = self.nav_field.shape[1], self.nav_field.shape[2]
            y = int(np.clip(pos[0], 0, H - 1))
            x = int(np.clip(pos[1], 0, W - 1))
            return self.nav_field[:, y, x].astype(np.float32)
        
        return np.zeros(2, dtype=np.float32)

    def __getitem__(self, idx):
        self._ensure_open()
        
        agent, t_idx = self.valid_indices[idx]
        agent = int(agent)
        t_idx = int(t_idx)

        pos_ds = self._fh["positions"]
        vel_ds = self._fh["velocities"] if self.has_vel else None
        
        # 获取目的地（如果存在）
        dest = None
        if self.destinations is not None:
            dest = int(self.destinations[t_idx, agent])

        # 历史 obs
        pos_hist = pos_ds[t_idx : t_idx + self.history, agent, :]  # (history,2)
        if vel_ds is not None:
            vel_hist = vel_ds[t_idx : t_idx + self.history, agent, :]  # (history,2)
        else:
            pos_hist_ext = pos_ds[t_idx : t_idx + self.history + 1, agent, :]
            vel_hist = np.diff(pos_hist_ext, axis=0)
        
        # 获取每个历史帧的导航方向（使用对应目的地的导航场）
        nav_hist = np.zeros((self.history, 2), dtype=np.float32)
        for i in range(self.history):
            nav_hist[i] = self._get_nav_direction(pos_hist[i], dest)

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
            "dest": dest if dest is not None else -1,
        }

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
