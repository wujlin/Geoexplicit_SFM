from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from src.phase3 import config


def load_environment(mask_path=None, score_path=None):
    """
    加载环境数据：
    - mask: 可行区域掩码
    - field_vec: 归一化的导航方向场 (2, H, W)，从 score_baseline.npz 读取
    - sdf: 到可行区域边界的距离场（正值在可行区内，用于墙壁斥力）
    """
    mask = np.load(mask_path or config.MASK_PATH)
    
    # 加载归一化的 score 向量场（方向场）
    score_path = score_path or config.SCORE_BASELINE_PATH
    score_data = np.load(score_path)
    score_y = score_data["score_y"]
    score_x = score_data["score_x"]
    field_vec = np.stack([score_y, score_x], axis=0)  # (2, H, W)
    
    print(f"[Environment] 加载 score 场: shape={field_vec.shape}, "
          f"magnitude range=[{np.hypot(score_y, score_x).min():.4f}, {np.hypot(score_y, score_x).max():.4f}]")

    # SDF 距离场（可行区域内为正值，表示到边界的距离）
    sdf = ndi.distance_transform_edt(mask)
    print(f"[Environment] SDF range: [{sdf.min():.2f}, {sdf.max():.2f}], walkable ratio: {mask.mean():.4f}")
    
    return mask.astype(np.float32), field_vec.astype(np.float32), sdf.astype(np.float32)
