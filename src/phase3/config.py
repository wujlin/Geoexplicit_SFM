"""
Phase 3 配置：物理参数与输出路径
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# 输入
MASK_PATH = BASE_DIR / "data" / "processed" / "walkable_mask.npy"
FIELD_BASELINE_PATH = BASE_DIR / "data" / "processed" / "field_baseline.npy"

# 输出
OUTPUT_DIR = BASE_DIR / "data" / "output"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"

# 物理/模拟参数
DT = 0.5
MAX_STEPS = 5000
NOISE_SIGMA = 0.2
TAU = 1.0
V0 = 1.0
AGENT_COUNT = 10000
RESPAWN_RADIUS = -1.0  # 此处设为负数避免提前终止，保持全程活跃

# 缓冲写入
BUFFER_STEPS = 1000
CHUNK_SIZE = 1000
