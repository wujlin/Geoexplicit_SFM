"""
Phase 3 物理模块 - 个体目的地版本（高性能）

关键优化：
1. 预加载所有 35 个导航场到内存 (~350MB)
2. 使用向量化 NumPy 操作代替 for 循环
3. 批量查询导航方向和距离
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from numba import njit, prange


class NavFieldCache:
    """
    预加载所有导航场到内存，支持批量查询
    """
    
    def __init__(self, nav_fields_dir: Path):
        """
        加载所有 35 个导航场
        
        Memory: ~350MB for 35 fields of shape (1365, 1435) × 3 arrays × float32
        """
        import json
        
        index_path = nav_fields_dir / "nav_fields_index.json"
        with open(index_path) as f:
            index = json.load(f)
        
        self.sink_ids = index["sink_ids"]
        self.num_sinks = len(self.sink_ids)
        H, W = index["shape"]
        
        # 预分配数组
        self.nav_y = np.zeros((self.num_sinks, H, W), dtype=np.float32)
        self.nav_x = np.zeros((self.num_sinks, H, W), dtype=np.float32)
        self.dist = np.zeros((self.num_sinks, H, W), dtype=np.float32)
        
        # sink_id -> 数组索引的映射
        self.sink_to_idx = {sid: i for i, sid in enumerate(self.sink_ids)}
        
        print(f"Loading {self.num_sinks} navigation fields into memory...")
        
        for i, sink_id in enumerate(self.sink_ids):
            path = nav_fields_dir / f"nav_field_{sink_id:03d}.npz"
            data = np.load(path)
            self.nav_y[i] = data["nav_y"]
            self.nav_x[i] = data["nav_x"]
            self.dist[i] = data["distance_field"]
        
        mem_mb = (self.nav_y.nbytes + self.nav_x.nbytes + self.dist.nbytes) / 1024 / 1024
        print(f"  Loaded! Memory usage: {mem_mb:.1f} MB")
        
    def get_nav_batch(self, dest: np.ndarray, py: np.ndarray, px: np.ndarray):
        """
        批量获取导航方向
        
        Args:
            dest: (N,) 目的地 sink ID
            py: (N,) y 坐标（整数）
            px: (N,) x 坐标（整数）
            
        Returns:
            nav_y: (N,) 导航 y 方向
            nav_x: (N,) 导航 x 方向
        """
        # 将 sink_id 转换为数组索引
        idx = np.array([self.sink_to_idx[d] for d in dest], dtype=np.int32)
        
        # 批量查询 (使用高级索引)
        nav_y = self.nav_y[idx, py, px]
        nav_x = self.nav_x[idx, py, px]
        
        return nav_y, nav_x
    
    def get_dist_batch(self, dest: np.ndarray, py: np.ndarray, px: np.ndarray):
        """
        批量获取到目的地的距离
        
        Args:
            dest: (N,) 目的地 sink ID
            py: (N,) y 坐标（整数）
            px: (N,) x 坐标（整数）
            
        Returns:
            distances: (N,) 距离
        """
        idx = np.array([self.sink_to_idx[d] for d in dest], dtype=np.int32)
        return self.dist[idx, py, px]


@njit(parallel=True, cache=True)
def _step_kernel_vectorized(
    pos: np.ndarray,        # (N, 2) float32
    vel: np.ndarray,        # (N, 2) float32
    active: np.ndarray,     # (N,) bool
    dest_idx: np.ndarray,   # (N,) int32 - 数组索引（不是 sink_id）
    nav_y_all: np.ndarray,  # (num_sinks, H, W) float32
    nav_x_all: np.ndarray,  # (num_sinks, H, W) float32
    dist_all: np.ndarray,   # (num_sinks, H, W) float32
    sdf: np.ndarray,        # (H, W) float32
    dt: float,
    noise_sigma: float,
    v0: float,
    wall_dist_thresh: float,
    wall_push_strength: float,
    off_road_recovery: float,
    momentum: float,
    arrival_threshold: float,
) -> np.ndarray:
    """
    Numba 并行化的物理更新内核
    
    Returns:
        arrived_mask: (N,) bool 到达标志
    """
    N = pos.shape[0]
    H, W = sdf.shape
    
    arrived_mask = np.zeros(N, dtype=np.bool_)
    
    for i in prange(N):
        if not active[i]:
            continue
        
        y, x = pos[i, 0], pos[i, 1]
        prev_vy, prev_vx = vel[i, 0], vel[i, 1]
        didx = dest_idx[i]
        
        # 边界 clamp
        if y < 1.0:
            y = 1.0
        if y > H - 2.0:
            y = H - 2.0
        if x < 1.0:
            x = 1.0
        if x > W - 2.0:
            x = W - 2.0
        
        old_y, old_x = y, x
        
        yi = int(y)
        xi = int(x)
        dist = sdf[yi, xi]
        is_on_road = dist > 0
        
        # === 检测到达 ===
        dist_to_dest = dist_all[didx, yi, xi]
        if dist_to_dest < arrival_threshold:
            arrived_mask[i] = True
            continue
        
        # === 1. 获取导航方向 ===
        nav_y_val = nav_y_all[didx, yi, xi]
        nav_x_val = nav_x_all[didx, yi, xi]
        nav_vy = nav_y_val * v0
        nav_vx = nav_x_val * v0
        
        # === 2. 墙壁斥力 / 掉网恢复 ===
        wall_vy, wall_vx = 0.0, 0.0
        
        # 计算 SDF 梯度
        yi_up = yi - 1 if yi > 0 else 0
        yi_dn = yi + 1 if yi < H - 1 else H - 1
        xi_lf = xi - 1 if xi > 0 else 0
        xi_rt = xi + 1 if xi < W - 1 else W - 1
        
        grad_y = (sdf[yi_dn, xi] - sdf[yi_up, xi]) * 0.5
        grad_x = (sdf[yi, xi_rt] - sdf[yi, xi_lf]) * 0.5
        grad_mag = np.sqrt(grad_y**2 + grad_x**2) + 1e-6
        
        if not is_on_road:
            wall_vy = (grad_y / grad_mag) * off_road_recovery
            wall_vx = (grad_x / grad_mag) * off_road_recovery
            nav_vy = 0.0
            nav_vx = 0.0
        elif dist < wall_dist_thresh:
            push = wall_push_strength * (1.0 - dist / wall_dist_thresh)
            wall_vy = (grad_y / grad_mag) * push
            wall_vx = (grad_x / grad_mag) * push
        
        # === 3. 噪声 ===
        ny = noise_sigma * np.random.randn()
        nx = noise_sigma * np.random.randn()
        
        # === 4. 目标速度 ===
        target_vy = nav_vy + wall_vy + ny
        target_vx = nav_vx + wall_vx + nx
        
        # === 5. 动量混合 ===
        eff_vy = momentum * prev_vy + (1.0 - momentum) * target_vy
        eff_vx = momentum * prev_vx + (1.0 - momentum) * target_vx
        
        # 速度限制
        speed = np.sqrt(eff_vy**2 + eff_vx**2)
        max_speed = v0 * 2.0
        if speed > max_speed:
            eff_vy = eff_vy / speed * max_speed
            eff_vx = eff_vx / speed * max_speed
        
        # 确保最小速度
        if speed < 0.1 and is_on_road:
            eff_vy = nav_vy
            eff_vx = nav_vx
        
        # 位置更新
        new_y = y + eff_vy * dt
        new_x = x + eff_vx * dt
        
        # 边界约束
        if new_y < 1.0:
            new_y = 1.0
        if new_y > H - 2.0:
            new_y = H - 2.0
        if new_x < 1.0:
            new_x = 1.0
        if new_x > W - 2.0:
            new_x = W - 2.0
        
        # === 6. 道路约束 ===
        new_yi = int(new_y)
        new_xi = int(new_x)
        if sdf[new_yi, new_xi] < 0:
            best_y, best_x = y, x
            best_dot = -2.0
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    cny = yi + dy
                    cnx = xi + dx
                    if 0 <= cny < H and 0 <= cnx < W:
                        if sdf[cny, cnx] > 0:
                            dot = dy * eff_vy + dx * eff_vx
                            if dot > best_dot:
                                best_dot = dot
                                best_y = float(cny)
                                best_x = float(cnx)
            
            new_y = best_y
            new_x = best_x
        
        pos[i, 0] = new_y
        pos[i, 1] = new_x
        
        # 计算实际 velocity
        vel[i, 0] = (new_y - old_y) / dt
        vel[i, 1] = (new_x - old_x) / dt
    
    return arrived_mask


class FastPhysicsEngine:
    """
    高性能物理引擎封装
    """
    
    def __init__(self, nav_fields_dir: Path, sdf: np.ndarray):
        """
        Args:
            nav_fields_dir: 导航场目录
            sdf: (H, W) 有符号距离场
        """
        self.cache = NavFieldCache(nav_fields_dir)
        self.sdf = sdf.astype(np.float32)
        
        # 创建 sink_id -> index 的反向映射
        self.sink_to_idx = self.cache.sink_to_idx
    
    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        active: np.ndarray,
        dest: np.ndarray,  # sink_id
        dt: float,
        noise_sigma: float,
        v0: float,
        wall_dist_thresh: float,
        wall_push_strength: float,
        off_road_recovery: float,
        momentum: float,
        arrival_threshold: float,
    ) -> np.ndarray:
        """
        执行一步物理更新
        
        Returns:
            arrived_indices: 到达的 agent 索引
        """
        # 将 sink_id 转换为数组索引
        dest_idx = np.array([self.sink_to_idx[int(d)] for d in dest], dtype=np.int32)
        
        arrived_mask = _step_kernel_vectorized(
            pos, vel, active, dest_idx,
            self.cache.nav_y, self.cache.nav_x, self.cache.dist,
            self.sdf,
            dt, noise_sigma, v0,
            wall_dist_thresh, wall_push_strength, off_road_recovery, momentum,
            arrival_threshold,
        )
        
        return np.where(arrived_mask)[0]
