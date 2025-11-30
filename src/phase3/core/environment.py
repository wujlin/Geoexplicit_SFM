from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from src.phase3 import config


def load_environment(mask_path=None, field_path=None):
    mask = np.load(mask_path or config.MASK_PATH)
    field = np.load(field_path or config.FIELD_BASELINE_PATH)  # shape (H,W) or (2,H,W)
    if field.ndim == 3:
        field_vec = field
    else:
        # 若仅有标量场，简单做梯度
        gy, gx = np.gradient(field)
        field_vec = np.stack([gy, gx], axis=0)

    # SDF 距离场（走路区域为1，其余为0）
    sdf = ndi.distance_transform_edt(mask)
    return mask.astype(np.float32), field_vec.astype(np.float32), sdf.astype(np.float32)
