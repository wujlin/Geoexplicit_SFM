"""
Phase 3 配置：物理参数与输出路径
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# 输入
MASK_PATH = BASE_DIR / "data" / "processed" / "walkable_mask.npy"
FIELD_BASELINE_PATH = BASE_DIR / "data" / "processed" / "field_baseline.npy"
SCORE_BASELINE_PATH = BASE_DIR / "data" / "processed" / "score_baseline.npz"  # 向量场
NAV_BASELINE_PATH = BASE_DIR / "data" / "processed" / "nav_baseline.npz"  # 距离场导航（优先）
TARGET_DENSITY_PATH = BASE_DIR / "data" / "processed" / "target_density.npy"

# 输出
OUTPUT_DIR = BASE_DIR / "data" / "output"
TRAJ_PATH = OUTPUT_DIR / "trajectories.h5"

# 物理/模拟参数
DT = 1.0                 # 时间步长
MAX_STEPS = 10000        # 总仿真步数
NOISE_SIGMA = 0.08       # 稍微增加噪声，使轨迹更自然
TAU = 0.5                # 未使用
V0 = 1.5                 # 速度：使用硬约束后可以提高
AGENT_COUNT = 10000      # 粒子数量

# 重生/边界参数
RESPAWN_RADIUS = -1.0    # 禁用基于SDF的respawn，仅靠到达终点
WALL_DIST_THRESH = 3.0   # 距边界多近开始斥力（像素）- 减小避免过度干扰
WALL_PUSH_STRENGTH = 1.5 # 墙壁斥力强度 - 减小
OFF_ROAD_RECOVERY = 3.0  # 掉网恢复推力 - 减小避免振荡
MOMENTUM = 0.7           # 动量系数 - 降低使加速更快

# 像素/秒缩放：若 V0 为米/秒，则 V0_PIX = V0 / GRID_RES_M
GRID_RES_M = 100.0       # 对应 Phase2 栅格分辨率

# 缓冲写入
BUFFER_STEPS = 1000
CHUNK_SIZE = 1000
