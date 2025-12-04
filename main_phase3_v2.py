"""
Phase 3 轨迹生成 - 个体目的地版本

核心改进：
1. 每个 agent 有独立的目的地（基于 OD 概率采样）
2. 使用对应目的地的导航场进行导航
3. 到达目的地后重生
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.phase3 import config
from src.phase3.simulation.spawner_v2 import IndividualDestSpawner
from src.phase3.simulation.engine_v2 import IndividualDestEngine
from src.phase3.simulation.recorder_v2 import TrajRecorderV2


def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """
    计算有符号距离场
    - 可行区内部为正值
    - 可行区外部为负值
    """
    mask_bool = mask > 0
    dist_inside = distance_transform_edt(mask_bool)
    dist_outside = distance_transform_edt(~mask_bool)
    sdf = dist_inside - dist_outside
    return sdf.astype(np.float32)


def main():
    print("=" * 60)
    print("Phase 3: Trajectory Generation (Individual Destination)")
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
    
    # Spawner（使用目标密度作为出发位置权重）
    spawner = IndividualDestSpawner(
        mask=mask,
        weight_map=target_density,  # 高流量区域更多出发
    )
    
    # Recorder
    recorder = TrajRecorderV2(agent_count=config.AGENT_COUNT)
    
    # Engine
    engine = IndividualDestEngine(
        mask=mask,
        sdf=sdf,
        spawner=spawner,
        recorder=recorder,
        arrival_threshold=10.0,  # 10 像素 = 1km
    )
    
    # 运行仿真
    print("\n[4/4] Running simulation...")
    print(f"  Agent count: {config.AGENT_COUNT}")
    print(f"  Max steps: {config.MAX_STEPS}")
    print(f"  V0: {config.V0}")
    print(f"  DT: {config.DT}")
    
    for step in range(config.MAX_STEPS):
        engine.step()
        
        if (step + 1) % 1000 == 0:
            stats = engine.get_stats()
            print(f"  Step {step+1}/{config.MAX_STEPS}: "
                  f"arrivals={stats['total_arrivals']}, "
                  f"rate={stats['arrival_rate']:.2f}/step")
    
    # 关闭记录器
    recorder.close()
    
    # 最终统计
    stats = engine.get_stats()
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"  Total steps: {stats['step_count']}")
    print(f"  Total arrivals: {stats['total_arrivals']}")
    print(f"  Arrival rate: {stats['arrival_rate']:.2f}/step")
    print(f"  Output: {config.TRAJ_PATH}")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    main()
