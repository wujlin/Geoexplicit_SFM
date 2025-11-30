"""
Phase 3 数据工厂入口：基于 Baseline 场生成大规模轨迹。
"""

from src.phase3 import config
from src.phase3.core.environment import load_environment
from src.phase3.simulation.engine import Engine
from src.phase3.simulation.recorder import TrajRecorder
from src.phase3.simulation.spawner import Spawner


def main():
    print("加载环境...")
    mask, field, sdf = load_environment()
    print("初始化组件...")
    spawner = Spawner(od_path=None, mask_shape=mask.shape)
    recorder = TrajRecorder(agent_count=config.AGENT_COUNT, buffer_steps=config.BUFFER_STEPS)
    engine = Engine(mask, field, sdf, spawner, recorder)

    print(f"开始模拟: agents={config.AGENT_COUNT}, steps={config.MAX_STEPS}")
    for step in range(config.MAX_STEPS):
        engine.step()
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{config.MAX_STEPS}")
    recorder.close()
    print(f"完成，轨迹写入: {config.TRAJ_PATH}")


if __name__ == "__main__":
    main()
