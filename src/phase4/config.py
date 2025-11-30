"""
Phase 4 配置：数据路径与窗口长度
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# 数据路径
TRAJ_H5_PATH = BASE_DIR / "data" / "output" / "trajectories.h5"

# 窗口配置
HISTORY_STEPS = 2
FUTURE_STEPS = 8
STRIDE = 1
