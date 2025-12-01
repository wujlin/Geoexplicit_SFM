"""
HDF5 轨迹数据集：滑动窗口读取 (observation, action)
- observation: 过去 history 步的位置与速度 (shape: history, 4)
- action: 未来 future 步的速度序列 (shape: future, 2)
- 过滤低速样本：只保留 action 平均速度 > min_speed 的样本
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
        min_speed: float = 0.1,  # 过滤掉平均速度小于此值的样本
    ):
        self.h5_path = Path(h5_path)
        self.history = history
        self.future = future
        self.stride = stride
        self.agent_ids = agent_ids
        self.min_speed = min_speed

        # 预计算有效样本索引
        self._precompute_valid_indices()

    def _precompute_valid_indices(self):
        """预计算所有满足速度阈值的样本索引（多进程加速）"""
        import multiprocessing as mp
        from tqdm import tqdm
        
        print(f"Precomputing valid indices with min_speed={self.min_speed}...")
        
        with h5py.File(self.h5_path, "r") as f:
            self.pos_shape = f["positions"].shape  # (T, N, 2)
            self.has_vel = "velocities" in f
            T, N, _ = self.pos_shape
            
            if self.agent_ids is None:
                self.agent_ids = np.arange(N, dtype=np.int64)
            else:
                self.agent_ids = np.asarray(self.agent_ids, dtype=np.int64)
            
            windows_per_agent = (T - (self.history + self.future)) // self.stride
            if windows_per_agent <= 0:
                raise ValueError("时间长度不足以构造样本")
            
            # 加载速度数据到内存（用于多进程共享）
            print("Loading velocity data into memory...")
            if self.has_vel:
                velocities = f["velocities"][:]  # (T, N, 2)
            else:
                positions = f["positions"][:]
                velocities = np.diff(positions, axis=0)
                velocities = np.concatenate([velocities, velocities[-1:]], axis=0)
        
        # 准备多进程参数
        num_agents = len(self.agent_ids)
        print(f"Processing {num_agents} agents × {windows_per_agent} windows...")
        
        # 使用多进程处理
        num_workers = mp.cpu_count()
        print(f"Using {num_workers} workers...")
        
        # 将 agent 分成多个 chunk
        chunk_size = max(1, num_agents // num_workers)
        agent_chunks = [
            self.agent_ids[i:i + chunk_size] 
            for i in range(0, num_agents, chunk_size)
        ]
        
        # 创建进程池
        from functools import partial
        worker_fn = partial(
            _filter_agents_chunk,
            velocities=velocities,
            history=self.history,
            future=self.future,
            stride=self.stride,
            windows_per_agent=windows_per_agent,
            min_speed=self.min_speed,
        )
        
        valid_indices = []
        with mp.Pool(num_workers) as pool:
            # 使用 imap 支持进度条
            results = list(tqdm(
                pool.imap(worker_fn, agent_chunks),
                total=len(agent_chunks),
                desc="Filtering samples"
            ))
        
        # 合并结果
        for chunk_result in results:
            valid_indices.extend(chunk_result)
        
        self.valid_indices = valid_indices
        self.total = len(valid_indices)
        
        total_possible = num_agents * windows_per_agent
        keep_ratio = self.total / total_possible * 100
        print(f"Valid samples: {self.total:,} / {total_possible:,} ({keep_ratio:.1f}%)")
        
        self._fh = None


def _filter_agents_chunk(agent_ids, velocities, history, future, stride, windows_per_agent, min_speed):
    """处理一批 agent 的过滤（供多进程调用）- 向量化加速"""
    valid_indices = []
    
    for agent in agent_ids:
        # 向量化处理该 agent 的所有窗口
        agent_vel = velocities[:, agent, :]  # (T, 2)
        
        for w in range(windows_per_agent):
            t_idx = w * stride
            # 获取 future 时间段的速度
            future_vel = agent_vel[t_idx + history : t_idx + history + future, :]
            avg_speed = np.linalg.norm(future_vel, axis=-1).mean()
            
            if avg_speed >= min_speed:
                valid_indices.append((int(agent), t_idx))
    
    return valid_indices

    def __len__(self):
        return self.total

    def _ensure_open(self):
        if self._fh is None:
            self._fh = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx):
        self._ensure_open()
        
        agent, t_idx = self.valid_indices[idx]

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
