from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True)
def bilinear_sample(field, y, x):
    """
    field: (2, H, W)
    y, x: scalar (float)
    返回 (2,) 双线性插值
    """
    h, w = field.shape[1], field.shape[2]
    # 边界clamp
    y = max(0.0, min(y, h - 1.001))
    x = max(0.0, min(x, w - 1.001))
    
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, h - 1)
    x1 = min(x0 + 1, w - 1)
    dy = y - y0
    dx = x - x0

    v00 = field[:, y0, x0]
    v01 = field[:, y0, x1]
    v10 = field[:, y1, x0]
    v11 = field[:, y1, x1]
    return (
        v00 * (1 - dx) * (1 - dy)
        + v01 * dx * (1 - dy)
        + v10 * (1 - dx) * dy
        + v11 * dx * dy
    )


@numba.jit(nopython=True, parallel=True)
def step_kernel(
    pos,
    vel,
    active,
    field,
    sdf,
    mask,
    dt,
    noise_sigma,
    v0,
    wall_dist_thresh,
    wall_push_strength,
    off_road_recovery,
):
    """
    过阻尼 Langevin 动力学，带墙壁斥力和掉网恢复。
    
    - field: (2, H, W) 归一化的导航方向场
    - sdf: 距离场（可行区正值）
    - mask: 可行区掩码
    - wall_dist_thresh: 离边界多近开始斥力
    - wall_push_strength: 斥力强度
    - off_road_recovery: 掉网时的恢复推力
    """
    N = pos.shape[0]
    H, W = sdf.shape
    
    for i in numba.prange(N):
        if not active[i]:
            continue
            
        y, x = pos[i, 0], pos[i, 1]
        
        # 边界 clamp（防止越界）
        y = max(1.0, min(y, H - 2.0))
        x = max(1.0, min(x, W - 2.0))
        
        yi = int(y)
        xi = int(x)
        dist = sdf[yi, xi]
        is_on_road = mask[yi, xi] > 0
        
        # === 1. 导航力 ===
        nav_dir = bilinear_sample(field, y, x)
        nav_vy = nav_dir[0] * v0
        nav_vx = nav_dir[1] * v0
        
        # === 2. 墙壁斥力 / 掉网恢复 ===
        wall_vy, wall_vx = 0.0, 0.0
        
        if not is_on_road:
            # 掉网：用 SDF 梯度找到最近可行区方向，强力推回
            grad_y = (sdf[min(yi+1, H-1), xi] - sdf[max(yi-1, 0), xi]) * 0.5
            grad_x = (sdf[yi, min(xi+1, W-1)] - sdf[yi, max(xi-1, 0)]) * 0.5
            grad_mag = np.sqrt(grad_y * grad_y + grad_x * grad_x) + 1e-6
            # 往 SDF 增大的方向走（回到路网）
            wall_vy = (grad_y / grad_mag) * off_road_recovery
            wall_vx = (grad_x / grad_mag) * off_road_recovery
            # 掉网时抑制导航力
            nav_vy *= 0.1
            nav_vx *= 0.1
            
        elif dist < wall_dist_thresh:
            # 靠近边界：柔性斥力
            grad_y = (sdf[min(yi+1, H-1), xi] - sdf[max(yi-1, 0), xi]) * 0.5
            grad_x = (sdf[yi, min(xi+1, W-1)] - sdf[yi, max(xi-1, 0)]) * 0.5
            grad_mag = np.sqrt(grad_y * grad_y + grad_x * grad_x) + 1e-6
            # 斥力随距离衰减
            push = wall_push_strength * (1.0 - dist / wall_dist_thresh)
            wall_vy = (grad_y / grad_mag) * push
            wall_vx = (grad_x / grad_mag) * push
        
        # === 3. 噪声 ===
        ny = noise_sigma * np.random.randn()
        nx = noise_sigma * np.random.randn()
        
        # === 4. 合成速度并积分 ===
        eff_vy = nav_vy + wall_vy + ny
        eff_vx = nav_vx + wall_vx + nx
        
        # 速度限制（防止过大跳跃）
        speed = np.sqrt(eff_vy * eff_vy + eff_vx * eff_vx)
        max_speed = v0 * 2.0
        if speed > max_speed:
            eff_vy = eff_vy / speed * max_speed
            eff_vx = eff_vx / speed * max_speed
        
        # 位置更新
        new_y = y + eff_vy * dt
        new_x = x + eff_vx * dt
        
        # 边界约束
        new_y = max(1.0, min(new_y, H - 2.0))
        new_x = max(1.0, min(new_x, W - 2.0))
        
        pos[i, 0] = new_y
        pos[i, 1] = new_x
        vel[i, 0] = eff_vy
        vel[i, 1] = eff_vx
    
    return 0
