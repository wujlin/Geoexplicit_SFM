"""
Phase 3 仿真器 - Spawner 模块（个体目的地版本）

每个 agent 有独立的：
1. 出发位置（基于可行区域采样）
2. 目的地 sink（基于 OD 概率分布）

OD 概率逻辑：
- 每个 agent 出生在某个 origin_sink 区域内
- 根据该 origin_sink 的 OD 概率分布采样目的地 dest_sink
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.phase3 import config


class IndividualDestSpawner:
    """
    支持个体目的地的 Spawner
    
    核心功能：
    1. sample_positions: 采样出发位置
    2. sample_destinations: 为每个 agent 采样目的地 sink
    3. respawn: 重置位置、速度和目的地
    """
    
    def __init__(
        self,
        mask: np.ndarray,
        weight_map: np.ndarray | None = None,
        tract_sink_mapping_path: Path | None = None,
        od_prob_path: Path | None = None,
        tract_pixel_mapping_path: Path | None = None,
    ):
        """
        Args:
            mask: (H, W) 可行走掩膜
            weight_map: (H, W) 可选的出发位置权重
            tract_sink_mapping_path: tract→sink 映射文件
            od_prob_path: sink-level OD 概率表
            tract_pixel_mapping_path: tract→pixel 映射文件
        """
        self.mask = mask.astype(np.float32)
        self.H, self.W = mask.shape
        
        # === 1. 构建出发位置概率 ===
        self._build_position_prob(weight_map)
        
        # === 2. 加载 OD 采样表 ===
        self._load_od_tables(tract_sink_mapping_path, od_prob_path, tract_pixel_mapping_path)
        
    def _build_position_prob(self, weight_map: np.ndarray | None):
        """构建出发位置的采样概率"""
        walkable = self.mask > 0
        
        if weight_map is not None:
            prob = weight_map.astype(np.float32).copy()
            prob[~walkable] = 0
        else:
            prob = walkable.astype(np.float32)
        
        prob[~walkable] = 0
        total = prob.sum()
        
        if total > 0:
            self.pos_prob = prob.ravel() / total
        else:
            self.pos_prob = walkable.ravel().astype(np.float32) / walkable.sum()
        
        # 为每个像素预计算它属于哪个 origin_sink
        # 这需要 tract_pixel_mapping 和 tract_sink_mapping
        self.pixel_to_sink = None  # 稍后在 _load_od_tables 中设置
        
    def _load_od_tables(
        self,
        tract_sink_mapping_path: Path | None,
        od_prob_path: Path | None,
        tract_pixel_mapping_path: Path | None,
    ):
        """加载 OD 采样所需的表"""
        base_dir = Path(config.BASE_DIR)
        
        if tract_sink_mapping_path is None:
            tract_sink_mapping_path = base_dir / "data" / "processed" / "tract_sink_mapping.csv"
        if od_prob_path is None:
            od_prob_path = base_dir / "data" / "processed" / "sink_od_prob.csv"
        if tract_pixel_mapping_path is None:
            tract_pixel_mapping_path = base_dir / "data" / "processed" / "tract_pixel_mapping.csv"
        
        # 检查文件是否存在
        if not tract_sink_mapping_path.exists():
            print(f"Warning: {tract_sink_mapping_path} not found. Using uniform destination sampling.")
            self.od_prob_table = None
            return
        
        # 加载 tract → sink 映射
        tract_sink = pd.read_csv(tract_sink_mapping_path)
        tract_pixel = pd.read_csv(tract_pixel_mapping_path)
        
        # 合并得到 pixel → sink 映射
        # 对于每个 tract，其质心像素 (px, py) 对应该 tract 所属的 sink
        tract_data = tract_pixel.merge(tract_sink[["GEOID", "sink_id"]], on="GEOID")
        
        # 为整个栅格创建 pixel → origin_sink 映射
        # 对于不在任何 tract 内的像素，使用最近邻 sink
        self._build_pixel_to_sink(tract_data)
        
        # 加载 OD 概率表
        if od_prob_path.exists():
            self.od_prob_table = pd.read_csv(od_prob_path)
            # 转换为字典以便快速查询
            # key: origin_sink, value: (dest_sinks, probs)
            self._build_od_sampling_dict()
        else:
            self.od_prob_table = None
            
    def _build_pixel_to_sink(self, tract_data: pd.DataFrame):
        """
        构建 pixel → origin_sink 映射
        
        策略：对于每个可行像素，找最近的 tract 质心，使用该 tract 的 sink
        """
        from scipy.spatial import cKDTree
        
        # 提取 tract 质心坐标和对应 sink
        tract_coords = tract_data[["px", "py"]].values
        tract_sinks = tract_data["sink_id"].values
        
        # 构建 KD-tree
        tree = cKDTree(tract_coords)
        
        # 为所有可行像素找最近的 tract
        walkable_indices = np.where(self.mask.ravel() > 0)[0]
        walkable_ys = walkable_indices // self.W
        walkable_xs = walkable_indices % self.W
        walkable_coords = np.stack([walkable_xs, walkable_ys], axis=1)  # (N, 2) as (px, py)
        
        # 查询最近邻
        _, indices = tree.query(walkable_coords, k=1)
        nearest_sinks = tract_sinks[indices]
        
        # 创建 pixel → sink 映射（只存储可行像素）
        # 使用扁平索引
        self.pixel_to_sink_flat = np.full(self.H * self.W, -1, dtype=np.int32)
        self.pixel_to_sink_flat[walkable_indices] = nearest_sinks
        
        print(f"  Pixel-to-sink mapping built: {len(walkable_indices)} walkable pixels")
        unique_sinks = np.unique(nearest_sinks)
        print(f"  Origin sinks represented: {len(unique_sinks)}")
        
    def _build_od_sampling_dict(self):
        """构建 OD 采样字典"""
        self.od_sampling = {}
        
        for origin_sink in self.od_prob_table["origin_sink"].unique():
            subset = self.od_prob_table[self.od_prob_table["origin_sink"] == origin_sink]
            dest_sinks = subset["dest_sink"].values.astype(np.int32)
            probs = subset["prob"].values.astype(np.float64)
            
            # 归一化（防止浮点误差）
            probs = probs / probs.sum()
            
            self.od_sampling[int(origin_sink)] = (dest_sinks, probs)
        
        print(f"  OD sampling dict built: {len(self.od_sampling)} origin sinks")
        
    def sample_positions(self, n: int) -> np.ndarray:
        """
        采样出发位置
        
        Args:
            n: 采样数量
            
        Returns:
            positions: (n, 2) 位置数组 [y, x]，带亚像素抖动
        """
        idx = np.random.choice(len(self.pos_prob), size=n, p=self.pos_prob)
        
        ys = idx // self.W
        xs = idx % self.W
        
        # 亚像素抖动
        ys = ys.astype(np.float32) + np.random.uniform(0.1, 0.9, size=n).astype(np.float32)
        xs = xs.astype(np.float32) + np.random.uniform(0.1, 0.9, size=n).astype(np.float32)
        
        return np.stack([ys, xs], axis=1)
    
    def sample_destinations(self, positions: np.ndarray) -> np.ndarray:
        """
        为每个位置采样目的地 sink
        
        Args:
            positions: (n, 2) 位置数组 [y, x]
            
        Returns:
            destinations: (n,) 目的地 sink ID 数组
        """
        n = len(positions)
        
        if self.od_prob_table is None or self.pixel_to_sink_flat is None:
            # 回退到均匀采样
            all_sinks = list(self.od_sampling.keys()) if self.od_sampling else list(range(35))
            return np.random.choice(all_sinks, size=n)
        
        destinations = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            # 获取位置的整数坐标
            yi = int(np.clip(positions[i, 0], 0, self.H - 1))
            xi = int(np.clip(positions[i, 1], 0, self.W - 1))
            flat_idx = yi * self.W + xi
            
            # 获取 origin_sink
            origin_sink = self.pixel_to_sink_flat[flat_idx]
            
            if origin_sink < 0 or origin_sink not in self.od_sampling:
                # 无效 origin，随机选择目的地
                all_sinks = list(self.od_sampling.keys())
                destinations[i] = np.random.choice(all_sinks) if all_sinks else 0
            else:
                # 从 OD 分布采样目的地
                dest_sinks, probs = self.od_sampling[origin_sink]
                destinations[i] = np.random.choice(dest_sinks, p=probs)
        
        return destinations
    
    def respawn(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        active: np.ndarray,
        dest: np.ndarray,
        indices: np.ndarray,
        nav_field_manager=None,
        v0: float = 1.0,
    ):
        """
        重置指定 agent 的位置、速度和目的地
        
        Args:
            pos: (N, 2) 位置数组
            vel: (N, 2) 速度数组
            active: (N,) 活跃标志数组
            dest: (N,) 目的地 sink ID 数组
            indices: 需要重置的 agent 索引
            nav_field_manager: 导航场管理器
            v0: 初始速度大小
        """
        if len(indices) == 0:
            return
        
        # 采样新位置
        new_pos = self.sample_positions(len(indices))
        
        # 采样新目的地
        new_dest = self.sample_destinations(new_pos)
        
        for i, idx in enumerate(indices):
            pos[idx, 0] = new_pos[i, 0]
            pos[idx, 1] = new_pos[i, 1]
            dest[idx] = new_dest[i]
            active[idx] = True
            
            # 初始化速度（使用对应目的地的导航场）
            if nav_field_manager is not None:
                yi = int(np.clip(new_pos[i, 0], 0, self.H - 1))
                xi = int(np.clip(new_pos[i, 1], 0, self.W - 1))
                
                nav_y, nav_x = nav_field_manager.get_nav_direction(new_dest[i], yi, xi)
                nav_mag = np.sqrt(nav_y**2 + nav_x**2) + 1e-6
                
                speed_factor = np.random.uniform(0.5, 1.0)
                vel[idx, 0] = (nav_y / nav_mag) * v0 * speed_factor
                vel[idx, 1] = (nav_x / nav_mag) * v0 * speed_factor
            else:
                # 无导航场时，随机方向
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0.3, 1.0) * v0
                vel[idx, 0] = np.cos(angle) * speed
                vel[idx, 1] = np.sin(angle) * speed


# 兼容旧版本的别名
Spawner = IndividualDestSpawner
