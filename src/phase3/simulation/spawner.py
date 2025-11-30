from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase3 import config


class Spawner:
    def __init__(self, mask: np.ndarray, weight_map: np.ndarray | None = None, od_path: str | None = None):
        self.mask = mask.astype(np.float32)
        self.weight_map = weight_map.astype(np.float32) if weight_map is not None else None
        self.od_path = od_path
        self.prob = None
        
        # 只在可行区域采样
        walkable = self.mask > 0
        if self.weight_map is not None:
            prob = self.weight_map.copy()
            prob[~walkable] = 0  # 非可行区域权重为 0
            prob = prob + 1e-12  # 避免全零
        else:
            prob = walkable.astype(np.float32)
        
        # 确保只有可行区域有非零概率
        prob[~walkable] = 0
        total = prob.sum()
        if total > 0:
            self.prob = prob.ravel() / total
        else:
            # 回退到均匀分布
            self.prob = walkable.ravel().astype(np.float32) / walkable.sum()
        
        if od_path:
            self._load_od(od_path)

    def _load_od(self, path):
        df = pd.read_csv(path)
        flow_col = [c for c in df.columns if c.lower() in ("s000", "total_jobs", "flow")]
        if flow_col:
            flow_col = flow_col[0]
        else:
            flow_col = df.columns[-1]
        df["prob"] = df[flow_col] / df[flow_col].sum()
        # 此处未将 tract 映射到栅格，保留以备扩展
        self.od_table = df

    def sample_positions(self, n):
        h, w = self.mask.shape
        if self.prob is None:
            # 不应该到这里，但以防万一
            walkable_idx = np.where(self.mask.ravel() > 0)[0]
            idx = np.random.choice(walkable_idx, size=n, replace=True)
        else:
            idx = np.random.choice(len(self.prob), size=n, p=self.prob)
        
        ys = idx // w
        xs = idx % w
        # 加入亚像素抖动（但保持在同一像素内）
        ys = ys + np.random.uniform(0.1, 0.9, size=n)
        xs = xs + np.random.uniform(0.1, 0.9, size=n)
        return np.stack([ys, xs], axis=1)

    def respawn(self, pos, vel, active, indices):
        """重置给定索引的粒子"""
        if len(indices) == 0:
            return
        new_pos = self.sample_positions(len(indices))
        for i, idx in enumerate(indices):
            pos[idx, 0] = new_pos[i, 0]
            pos[idx, 1] = new_pos[i, 1]
            vel[idx, :] = 0.0
            active[idx] = True
