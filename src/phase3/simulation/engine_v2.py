"""
Phase 3 仿真器 - Engine 模块（个体目的地版本）

核心改动：
1. 每个 agent 有独立的目的地 sink
2. 使用对应目的地的导航场进行导航
3. 支持到达终止条件
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.phase3 import config
from src.phase3.core.physics_v2 import step_kernel_individual_dest
from src.phase2.nav_field_generator import NavFieldManager


class IndividualDestEngine:
    """
    支持个体目的地的仿真引擎
    
    每个 agent 有：
    - pos: (y, x) 位置
    - vel: (vy, vx) 速度
    - dest: 目的地 sink ID
    - active: 是否活跃
    - arrived: 是否已到达目的地
    """
    
    def __init__(
        self,
        mask: np.ndarray,
        sdf: np.ndarray,
        spawner,
        recorder,
        nav_field_manager: NavFieldManager = None,
        arrival_threshold: float = 10.0,  # 到达阈值（像素）
    ):
        """
        Args:
            mask: (H, W) 可行走掩膜
            sdf: (H, W) 有符号距离场
            spawner: 个体目的地版本的 Spawner
            recorder: 轨迹记录器
            nav_field_manager: 导航场管理器
            arrival_threshold: 到达阈值（像素单位）
        """
        self.mask = mask
        self.sdf = sdf
        self.spawner = spawner
        self.recorder = recorder
        self.arrival_threshold = arrival_threshold
        
        # 加载导航场管理器
        if nav_field_manager is None:
            nav_fields_dir = Path(config.BASE_DIR) / "data" / "processed" / "nav_fields"
            if nav_fields_dir.exists():
                self.nav_field_manager = NavFieldManager(nav_fields_dir)
            else:
                raise FileNotFoundError(f"Nav fields directory not found: {nav_fields_dir}")
        else:
            self.nav_field_manager = nav_field_manager
        
        # 初始化 agent 状态
        n = config.AGENT_COUNT
        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.dest = np.zeros(n, dtype=np.int32)
        self.active = np.zeros(n, dtype=np.bool_)
        self.arrived = np.zeros(n, dtype=np.bool_)
        
        # 统计
        self.total_arrivals = 0
        self.step_count = 0
        
        # 初始化所有 agent
        self.spawner.respawn(
            self.pos, self.vel, self.active, self.dest,
            np.arange(n),
            nav_field_manager=self.nav_field_manager,
            v0=config.V0,
        )
        
        print(f"Engine initialized: {n} agents")
        dest_counts = np.bincount(self.dest, minlength=35)
        print(f"  Destination distribution: {dest_counts}")
        
    def step(self):
        """执行一步仿真"""
        self.step_count += 1
        
        # 调用物理内核
        arrived_indices = step_kernel_individual_dest(
            self.pos,
            self.vel,
            self.active,
            self.dest,
            self.arrived,
            self.sdf,
            self.mask,
            self.nav_field_manager,
            config.DT,
            config.NOISE_SIGMA,
            config.V0,
            config.WALL_DIST_THRESH,
            config.WALL_PUSH_STRENGTH,
            config.OFF_ROAD_RECOVERY,
            config.MOMENTUM,
            self.arrival_threshold,
        )
        
        # 处理到达的 agent
        if len(arrived_indices) > 0:
            self.total_arrivals += len(arrived_indices)
            
            # 标记为已到达
            for idx in arrived_indices:
                self.arrived[idx] = True
                self.active[idx] = False
            
            # 重生新的 agent
            self.spawner.respawn(
                self.pos, self.vel, self.active, self.dest,
                np.array(arrived_indices, dtype=np.int64),
                nav_field_manager=self.nav_field_manager,
                v0=config.V0,
            )
        
        # 记录轨迹
        self.recorder.collect(self.pos, self.vel, self.dest)
        
    def get_stats(self):
        """获取统计信息"""
        return {
            "step_count": self.step_count,
            "total_arrivals": self.total_arrivals,
            "active_count": self.active.sum(),
            "arrival_rate": self.total_arrivals / max(1, self.step_count),
        }


# 兼容旧版本
Engine = IndividualDestEngine
