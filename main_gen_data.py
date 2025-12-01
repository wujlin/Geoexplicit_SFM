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
    
    # 加载距离场用于初始化粒子位置
    dist_field_path = config.OUTPUT_DIR.parent / "processed" / "distance_field.npy"
    if dist_field_path.exists():
        dist_field = np.load(dist_field_path)
        # 只在可行区域采样，用距离场作为权重
        # 距离越远（离 sink 越远）越可能被选中
        # 过滤掉：1) sink 区域 (dist=0), 2) 异常孤立像素 (dist >= 5000)
        walkable = mask > 0
        valid = walkable & (dist_field > 0) & (dist_field < 5000)
        spawn_weight = np.zeros_like(dist_field)
        spawn_weight[valid] = np.clip(dist_field[valid], 10, 500)
        print(f"[Spawner] 使用距离场权重: 有效点 {valid.sum()}/{walkable.sum()}, "
              f"range=[{spawn_weight[valid].min():.0f}, {spawn_weight[valid].max():.0f}]")
    else:
        # 回退：均匀分布
        spawn_weight = (mask > 0).astype(np.float32)
        print("[Spawner] 均匀分布")

    n_agents = n_agents or config.AGENT_COUNT
    n_steps = n_steps or config.MAX_STEPS

    # 初始化粒子
    spawner = Spawner(mask=mask, weight_map=spawn_weight)
    pos = np.zeros((n_agents, 2), dtype=np.float32)
    vel = np.zeros((n_agents, 2), dtype=np.float32)
    active = np.zeros((n_agents,), dtype=np.bool_)
    
    # 使用 respawn 初始化，给予初始速度
    spawner.respawn(pos, vel, active, np.arange(n_agents), nav_field=field, v0=config.V0)
    
    # 打印初始距离和速度统计
    if dist_field_path.exists():
        H, W = mask.shape
        init_dists = [dist_field[int(np.clip(pos[i,0], 0, H-1)), 
                                  int(np.clip(pos[i,1], 0, W-1))] 
                      for i in range(min(100, n_agents))]
        print(f"[Init] 初始距离: mean={np.mean(init_dists):.0f}, min={np.min(init_dists):.0f}, max={np.max(init_dists):.0f}")
    
    init_speeds = np.sqrt((vel**2).sum(axis=1))
    print(f"[Init] 初始速度: mean={init_speeds.mean():.3f}, min={init_speeds.min():.3f}, max={init_speeds.max():.3f}")

    recorder = TrajRecorder(agent_count=n_agents, buffer_steps=config.BUFFER_STEPS, out_path=config.TRAJ_PATH)

    print(f"开始模拟: agents={n_agents}, steps={n_steps}")
    print(f"参数: V0={config.V0}, DT={config.DT}, NOISE={config.NOISE_SIGMA}, "
          f"WALL_THRESH={config.WALL_DIST_THRESH}, WALL_PUSH={config.WALL_PUSH_STRENGTH}, "
          f"OFF_ROAD_RECOVERY={config.OFF_ROAD_RECOVERY}")
    
    t0 = time.time()
    respawn_count = 0
    H, W = mask.shape
    
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
            config.MOMENTUM,
        )
        
        # 检查到达 sink 的 agent（距离 < 5 且速度 < 0.1）并重生
        if dist_field_path.exists():
            speed = np.sqrt((vel**2).sum(axis=1))
            arrived = []
            for i in range(n_agents):
                y = int(np.clip(pos[i, 0], 0, H-1))
                x = int(np.clip(pos[i, 1], 0, W-1))
                if dist_field[y, x] < 5 and speed[i] < 0.1:
                    arrived.append(i)
            
            if arrived:
                spawner.respawn(pos, vel, active, np.array(arrived), nav_field=field, v0=config.V0)
                respawn_count += len(arrived)

        recorder.collect(pos, vel)
        
        if (t + 1) % 500 == 0:
            elapsed = time.time() - t0
            steps_per_sec = (t + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_steps - t - 1) / steps_per_sec if steps_per_sec > 0 else -1
            eta_str = f"{eta/60:.1f} min" if eta > 0 else "N/A"
            # 统计当前速度
            speed = np.sqrt((vel**2).sum(axis=1))
            on_road_count = 0
            for i in range(min(1000, n_agents)):
                y = int(np.clip(pos[i, 0], 0, H-1))
                x = int(np.clip(pos[i, 1], 0, W-1))
                if mask[y, x] > 0:
                    on_road_count += 1
            on_road_ratio = on_road_count / min(1000, n_agents)
            print(f"step {t+1}/{n_steps} | elapsed {elapsed:.1f}s | {steps_per_sec:.2f} steps/s | ETA {eta_str} | "
                  f"speed mean={speed.mean():.3f} max={speed.max():.3f} | on_road={on_road_ratio*100:.1f}% | respawns={respawn_count}")

    recorder.close()
    total_time = time.time() - t0
    print(f"完成，轨迹写入: {config.TRAJ_PATH} | 总耗时 {total_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=None, help="number of agents")
    parser.add_argument("--steps", type=int, default=None, help="number of steps")
    args = parser.parse_args()
    main(n_agents=args.agents, n_steps=args.steps)
