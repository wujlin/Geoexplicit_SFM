"""
Phase 3 轨迹记录器 - 个体目的地版本

增加记录目的地 sink ID。
"""

from __future__ import annotations

import h5py
import numpy as np

from src.phase3 import config


class TrajRecorderV2:
    """
    轨迹记录器（支持目的地记录）
    
    记录内容：
    - positions: (T, N, 2) 位置 [y, x]
    - velocities: (T, N, 2) 速度 [vy, vx]
    - destinations: (T, N) 目的地 sink ID
    """
    
    def __init__(
        self,
        agent_count: int,
        buffer_steps: int = None,
        out_path: str = None,
    ):
        self.agent_count = agent_count
        self.buffer_steps = buffer_steps or config.BUFFER_STEPS
        self.out_path = out_path or config.TRAJ_PATH
        
        # 缓冲区
        self.buffer_pos = np.zeros((self.buffer_steps, agent_count, 2), dtype=np.float32)
        self.buffer_vel = np.zeros((self.buffer_steps, agent_count, 2), dtype=np.float32)
        self.buffer_dest = np.zeros((self.buffer_steps, agent_count), dtype=np.int32)
        
        self.ptr = 0
        self.file = None
        self._init_file()
        
    def _init_file(self):
        """初始化 HDF5 文件"""
        self.file = h5py.File(self.out_path, "w")
        
        self.dset_pos = self.file.create_dataset(
            "positions",
            shape=(0, self.agent_count, 2),
            maxshape=(None, self.agent_count, 2),
            dtype="f4",
            chunks=(min(1000, self.buffer_steps), self.agent_count, 2),
        )
        
        self.dset_vel = self.file.create_dataset(
            "velocities",
            shape=(0, self.agent_count, 2),
            maxshape=(None, self.agent_count, 2),
            dtype="f4",
            chunks=(min(1000, self.buffer_steps), self.agent_count, 2),
        )
        
        self.dset_dest = self.file.create_dataset(
            "destinations",
            shape=(0, self.agent_count),
            maxshape=(None, self.agent_count),
            dtype="i4",
            chunks=(min(1000, self.buffer_steps), self.agent_count),
        )
        
    def collect(self, pos: np.ndarray, vel: np.ndarray, dest: np.ndarray = None):
        """
        收集一帧数据
        
        Args:
            pos: (N, 2) 位置
            vel: (N, 2) 速度
            dest: (N,) 目的地 sink ID（可选，兼容旧版）
        """
        self.buffer_pos[self.ptr] = pos
        self.buffer_vel[self.ptr] = vel
        
        if dest is not None:
            self.buffer_dest[self.ptr] = dest
        
        self.ptr += 1
        
        if self.ptr >= self.buffer_steps:
            self._flush()
            
    def _flush(self):
        """刷新缓冲区到文件"""
        if self.ptr == 0:
            return
        
        n_old = self.dset_pos.shape[0]
        n_new = n_old + self.ptr
        
        self.dset_pos.resize((n_new, self.agent_count, 2))
        self.dset_vel.resize((n_new, self.agent_count, 2))
        self.dset_dest.resize((n_new, self.agent_count))
        
        self.dset_pos[n_old:n_new] = self.buffer_pos[:self.ptr]
        self.dset_vel[n_old:n_new] = self.buffer_vel[:self.ptr]
        self.dset_dest[n_old:n_new] = self.buffer_dest[:self.ptr]
        
        self.ptr = 0
        
    def close(self):
        """关闭文件"""
        self._flush()
        if self.file:
            self.file.close()
            self.file = None


# 兼容旧版本
TrajRecorder = TrajRecorderV2
