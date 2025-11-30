from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from src.phase3 import config


def load_environment(mask_path=None, nav_path=None):
    """
    加载环境数据：
    - mask: 可行区域掩码
    - field_vec: 归一化的导航方向场 (2, H, W)
    - sdf: 有符号距离场（可行区正值，非可行区负值）
    
    优先使用 nav_baseline.npz（基于距离场），否则回退到 score_baseline.npz
    """
    mask = np.load(mask_path or config.MASK_PATH)
    
    # 优先加载基于距离场的导航方向
    nav_path = config.NAV_BASELINE_PATH if hasattr(config, 'NAV_BASELINE_PATH') else None
    if nav_path and nav_path.exists():
        nav_data = np.load(nav_path)
        nav_y = nav_data["nav_y"]
        nav_x = nav_data["nav_x"]
        field_vec = np.stack([nav_y, nav_x], axis=0)
        field_type = "nav (distance-based)"
    else:
        # 回退到 score 场
        score_path = config.SCORE_BASELINE_PATH
        score_data = np.load(score_path)
        score_y = score_data["score_y"]
        score_x = score_data["score_x"]
        field_vec = np.stack([score_y, score_x], axis=0)
        field_type = "score (diffusion-based)"
    
    mag = np.hypot(field_vec[0], field_vec[1])
    print(f"[Environment] 加载 {field_type}: shape={field_vec.shape}, "
          f"magnitude range=[{mag.min():.4f}, {mag.max():.4f}], "
          f"mean={mag.mean():.4f}, >0.5 ratio={(mag > 0.5).mean()*100:.1f}%")

    # **有符号距离场 (Signed Distance Field)**
    # 可行区内：到边界的正距离
    # 可行区外：到边界的负距离
    mask_bool = mask > 0
    sdf_inside = ndi.distance_transform_edt(mask_bool)   # 可行区内到边界的距离
    sdf_outside = ndi.distance_transform_edt(~mask_bool)  # 非可行区到边界的距离
    sdf = sdf_inside - sdf_outside  # 可行区内正，可行区外负
    
    print(f"[Environment] SDF range: [{sdf.min():.2f}, {sdf.max():.2f}], walkable ratio: {mask.mean():.4f}")
    
    return mask.astype(np.float32), field_vec.astype(np.float32), sdf.astype(np.float32)
