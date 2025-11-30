"""
Phase 3 数据生成入口：无渲染模拟，输出 npz 轨迹。
"""

import argparse
import numpy as np
import time

from src.phase3 import config
from src.phase3.core.environment import load_environment
from src.phase3.core.physics import step_kernel
from src.phase3.simulation.spawner import Spawner
from src.phase3.simulation.recorder import TrajRecorder


def main(n_agents=None, n_steps=None):
    print("加载环境...")
    mask, field, sdf = load_environment()
    density = np.load(config.TARGET_DENSITY_PATH)

    n_agents = n_agents or config.AGENT_COUNT
    n_steps = n_steps or config.MAX_STEPS

    # 初始化粒子
    spawner = Spawner(mask=mask, weight_map=density)
    pos = spawner.sample_positions(n_agents).astype(np.float32)
    vel = np.zeros((n_agents, 2), dtype=np.float32)
    active = np.ones((n_agents,), dtype=np.bool_)

    recorder = TrajRecorder(agent_count=n_agents, buffer_steps=config.BUFFER_STEPS, out_path=config.TRAJ_PATH)

    print(f"开始模拟: agents={n_agents}, steps={n_steps}")
    print(f"参数: V0={config.V0}, DT={config.DT}, NOISE={config.NOISE_SIGMA}, "
          f"WALL_THRESH={config.WALL_DIST_THRESH}, WALL_PUSH={config.WALL_PUSH_STRENGTH}, "
          f"OFF_ROAD_RECOVERY={config.OFF_ROAD_RECOVERY}")
    
    t0 = time.time()
    for t in range(n_steps):
        step_kernel(
            pos,
            vel,
            active,
            field,
            sdf,
            mask,
            config.DT,
            config.NOISE_SIGMA,
            config.V0,
            config.WALL_DIST_THRESH,
            config.WALL_PUSH_STRENGTH,
            config.OFF_ROAD_RECOVERY,
        )

        recorder.collect(pos, vel)
        
        if (t + 1) % 500 == 0:
            elapsed = time.time() - t0
            steps_per_sec = (t + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_steps - t - 1) / steps_per_sec if steps_per_sec > 0 else -1
            eta_str = f"{eta/60:.1f} min" if eta > 0 else "N/A"
            # 统计当前速度
            speed = np.sqrt((vel**2).sum(axis=1))
            on_road = np.array([mask[int(pos[i,0]), int(pos[i,1])] > 0 for i in range(min(1000, n_agents))])
            print(f"step {t+1}/{n_steps} | elapsed {elapsed:.1f}s | {steps_per_sec:.2f} steps/s | ETA {eta_str} | "
                  f"speed mean={speed.mean():.3f} max={speed.max():.3f} | on_road={on_road.mean()*100:.1f}%")

    recorder.close()
    total_time = time.time() - t0
    print(f"完成，轨迹写入: {config.TRAJ_PATH} | 总耗时 {total_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=None, help="number of agents")
    parser.add_argument("--steps", type=int, default=None, help="number of steps")
    args = parser.parse_args()
    main(n_agents=args.agents, n_steps=args.steps)
