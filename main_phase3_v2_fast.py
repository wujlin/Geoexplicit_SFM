"""
Phase 3 轨迹生成 - 个体目的地版本（高性能）

关键优化：
1. 预加载所有 35 个导航场到内存
2. 使用 Numba 并行化物理内核
3. 预估运行时间：~5-10 分钟（对比原版 10+ 小时）
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.phase3 import config
from src.phase3.simulation.spawner_v2 import IndividualDestSpawner
from src.phase3.simulation.recorder_v2 import TrajRecorderV2
from src.phase3.core.physics_v2_fast import FastPhysicsEngine


def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """计算有符号距离场"""
    mask_bool = mask > 0
    dist_inside = distance_transform_edt(mask_bool)
    dist_outside = distance_transform_edt(~mask_bool)
    sdf = dist_inside - dist_outside
    return sdf.astype(np.float32)


class FastEngine:
    """高性能仿真引擎"""
    
    def __init__(
        self,
        mask: np.ndarray,
        sdf: np.ndarray,
        spawner,
        recorder,
        nav_fields_dir: Path,
        arrival_threshold: float = 10.0,
    ):
        self.mask = mask
        self.sdf = sdf
        self.spawner = spawner
        self.recorder = recorder
        self.arrival_threshold = arrival_threshold
        
        # 高性能物理引擎
        self.physics = FastPhysicsEngine(nav_fields_dir, sdf)
        
        # 初始化 agent 状态
        n = config.AGENT_COUNT
        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.dest = np.zeros(n, dtype=np.int32)
        self.active = np.zeros(n, dtype=np.bool_)
        
        # 统计
        self.total_arrivals = 0
        self.step_count = 0
        
        # 创建一个简单的 nav_field_manager 兼容接口
        class NavFieldManagerCompat:
            def __init__(self, cache):
                self.cache = cache
                self.sink_ids = set(cache.sink_to_idx.keys())
            
            def get_nav_direction(self, sink_id, py, px):
                idx = self.cache.sink_to_idx[int(sink_id)]
                py = np.clip(py, 0, self.cache.nav_y.shape[1] - 1)
                px = np.clip(px, 0, self.cache.nav_y.shape[2] - 1)
                return self.cache.nav_y[idx, py, px], self.cache.nav_x[idx, py, px]
        
        nav_compat = NavFieldManagerCompat(self.physics.cache)
        
        # 初始化所有 agent
        self.spawner.respawn(
            self.pos, self.vel, self.active, self.dest,
            np.arange(n),
            nav_field_manager=nav_compat,
            v0=config.V0,
        )
        
        print(f"FastEngine initialized: {n} agents")
        dest_counts = np.bincount(self.dest, minlength=35)
        print(f"  Destination distribution: {dest_counts}")
    
    def step(self):
        """执行一步仿真"""
        self.step_count += 1
        
        # 调用高性能物理内核
        arrived_indices = self.physics.step(
            self.pos, self.vel, self.active, self.dest,
            config.DT, config.NOISE_SIGMA, config.V0,
            config.WALL_DIST_THRESH, config.WALL_PUSH_STRENGTH,
            config.OFF_ROAD_RECOVERY, config.MOMENTUM,
            self.arrival_threshold,
        )
        
        # 处理到达的 agent
        if len(arrived_indices) > 0:
            self.total_arrivals += len(arrived_indices)
            
            # 标记为不活跃
            self.active[arrived_indices] = False
            
            # 创建兼容接口
            class NavFieldManagerCompat:
                def __init__(self, cache):
                    self.cache = cache
                    self.sink_ids = set(cache.sink_to_idx.keys())
                
                def get_nav_direction(self, sink_id, py, px):
                    idx = self.cache.sink_to_idx[int(sink_id)]
                    py = np.clip(py, 0, self.cache.nav_y.shape[1] - 1)
                    px = np.clip(px, 0, self.cache.nav_y.shape[2] - 1)
                    return self.cache.nav_y[idx, py, px], self.cache.nav_x[idx, py, px]
            
            nav_compat = NavFieldManagerCompat(self.physics.cache)
            
            # 重生
            self.spawner.respawn(
                self.pos, self.vel, self.active, self.dest,
                arrived_indices,
                nav_field_manager=nav_compat,
                v0=config.V0,
            )
        
        # 记录轨迹
        self.recorder.collect(self.pos, self.vel, self.dest)
    
    def get_stats(self):
        return {
            "step_count": self.step_count,
            "total_arrivals": self.total_arrivals,
            "active_count": self.active.sum(),
            "arrival_rate": self.total_arrivals / max(1, self.step_count),
        }


def main():
    print("=" * 60)
    print("Phase 3: Trajectory Generation (Fast Version)")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] Loading data...")
    mask = np.load(config.MASK_PATH)
    target_density = np.load(config.TARGET_DENSITY_PATH)
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Walkable pixels: {(mask > 0).sum():,}")
    
    # 计算 SDF
    print("\n[2/4] Computing SDF...")
    sdf = compute_sdf(mask)
    print(f"  SDF range: [{sdf.min():.1f}, {sdf.max():.1f}]")
    
    # 初始化组件
    print("\n[3/4] Initializing simulation...")
    
    # Spawner
    spawner = IndividualDestSpawner(
        mask=mask,
        weight_map=target_density,
    )
    
    # Recorder
    recorder = TrajRecorderV2(agent_count=config.AGENT_COUNT)
    
    # 导航场目录
    nav_fields_dir = Path(config.BASE_DIR) / "data" / "processed" / "nav_fields"
    
    # Engine
    engine = FastEngine(
        mask=mask,
        sdf=sdf,
        spawner=spawner,
        recorder=recorder,
        nav_fields_dir=nav_fields_dir,
        arrival_threshold=10.0,
    )
    
    # 运行仿真
    print("\n[4/4] Running simulation...")
    print(f"  Agent count: {config.AGENT_COUNT}")
    print(f"  Max steps: {config.MAX_STEPS}")
    print(f"  V0: {config.V0}")
    print(f"  DT: {config.DT}")
    
    import time
    start_time = time.time()
    
    for step in range(config.MAX_STEPS):
        engine.step()
        
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (step + 1) * (config.MAX_STEPS - step - 1)
            stats = engine.get_stats()
            print(f"  Step {step+1}/{config.MAX_STEPS}: "
                  f"arrivals={stats['total_arrivals']}, "
                  f"rate={stats['arrival_rate']:.2f}/step, "
                  f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s")
    
    # 关闭记录器
    recorder.close()
    
    # 最终统计
    total_time = time.time() - start_time
    stats = engine.get_stats()
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Total steps: {stats['step_count']}")
    print(f"  Total arrivals: {stats['total_arrivals']}")
    print(f"  Arrival rate: {stats['arrival_rate']:.2f}/step")
    print(f"  Output: {config.TRAJ_PATH}")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    main()
