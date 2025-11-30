from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase3 import config


class Spawner:
    def __init__(self, od_path=None, mask_shape=None):
        self.od_path = od_path
        self.mask_shape = mask_shape
        self.od_table = None
        if od_path:
            self._load_od(od_path)

    def _load_od(self, path):
        df = pd.read_csv(path)
        # 以工作地为重点，简单按总流量采样起点；可按需扩展
        flow_col = [c for c in df.columns if c.lower() in ("s000", "total_jobs", "flow")]
        if flow_col:
            flow_col = flow_col[0]
        else:
            flow_col = df.columns[-1]
        df["prob"] = df[flow_col] / df[flow_col].sum()
        self.od_table = df

    def sample_positions(self, n):
        if self.od_table is None or self.mask_shape is None:
            # 随机均匀采样
            h, w = self.mask_shape
            ys = np.random.uniform(0, h, size=n)
            xs = np.random.uniform(0, w, size=n)
            return np.stack([ys, xs], axis=1)
        idx = np.random.choice(len(self.od_table), size=n, p=self.od_table["prob"].values)
        # 暂用随机扰动，未用精确多边形
        h, w = self.mask_shape
        ys = np.random.uniform(0, h, size=n)
        xs = np.random.uniform(0, w, size=n)
        return np.stack([ys, xs], axis=1)

    def respawn(self, pos, vel, active, indices):
        """重置给定索引的粒子"""
        new_pos = self.sample_positions(len(indices))
        for i, idx in enumerate(indices):
            pos[idx, 0] = new_pos[i, 0]
            pos[idx, 1] = new_pos[i, 1]
            vel[idx, :] = 0.0
            active[idx] = True
