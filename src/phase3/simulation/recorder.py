from __future__ import annotations

import h5py
import numpy as np

from src.phase3 import config


class TrajRecorder:
    def __init__(self, agent_count, buffer_steps=config.BUFFER_STEPS, out_path=None):
        self.agent_count = agent_count
        self.buffer_steps = buffer_steps
        self.out_path = out_path or config.TRAJ_PATH
        self.buffer_pos = np.zeros((buffer_steps, agent_count, 2), dtype=np.float32)
        self.buffer_vel = np.zeros((buffer_steps, agent_count, 2), dtype=np.float32)
        self.ptr = 0
        self.file = None
        self._init_file()

    def _init_file(self):
        self.file = h5py.File(self.out_path, "w")
        self.dset_pos = self.file.create_dataset(
            "positions", shape=(0, self.agent_count, 2), maxshape=(None, self.agent_count, 2), dtype="f4", chunks=True
        )
        self.dset_vel = self.file.create_dataset(
            "velocities", shape=(0, self.agent_count, 2), maxshape=(None, self.agent_count, 2), dtype="f4", chunks=True
        )

    def collect(self, pos, vel):
        self.buffer_pos[self.ptr] = pos
        self.buffer_vel[self.ptr] = vel
        self.ptr += 1
        if self.ptr >= self.buffer_steps:
            self._flush()

    def _flush(self):
        if self.ptr == 0:
            return
        n_old = self.dset_pos.shape[0]
        n_new = n_old + self.ptr
        self.dset_pos.resize((n_new, self.agent_count, 2))
        self.dset_vel.resize((n_new, self.agent_count, 2))
        self.dset_pos[n_old:n_new] = self.buffer_pos[: self.ptr]
        self.dset_vel[n_old:n_new] = self.buffer_vel[: self.ptr]
        self.ptr = 0

    def close(self):
        self._flush()
        if self.file:
            self.file.close()
