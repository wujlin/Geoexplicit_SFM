from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True)
def bilinear_sample(field, y, x):
    """
    field: (2, H, W)
    y, x: scalar (float)
    返回 (2,) 最近邻采样（不使用双线性插值，因为导航场在窄道路上方向变化剧烈）
    """
    h, w = field.shape[1], field.shape[2]
    # 最近邻
    yi = int(round(y))
    xi = int(round(x))
    yi = max(0, min(yi, h - 1))
    xi = max(0, min(xi, w - 1))
    return field[:, yi, xi].copy()


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
    momentum=0.7,
):
    """
    过阻尼 Langevin 动力学，带墙壁斥力、掉网恢复和动量。
    
    - field: (2, H, W) 归一化的导航方向场
    - sdf: 有符号距离场（可行区正值，非可行区负值）
    - mask: 可行区掩码
    - wall_dist_thresh: 离边界多近开始斥力
    - wall_push_strength: 斥力强度
    - off_road_recovery: 掉网时的恢复推力
    - momentum: 动量系数（0=无惯性，1=完全惯性）
    
    注意：velocity 存储的是「实际位移 / dt」，而非「目标速度」。
    这确保了 velocity 与位置变化一致，便于 Diffusion 模型学习。
    """
    N = pos.shape[0]
    H, W = sdf.shape
    
    for i in numba.prange(N):
        if not active[i]:
            continue
            
        y, x = pos[i, 0], pos[i, 1]
        prev_vy, prev_vx = vel[i, 0], vel[i, 1]
        
        # 边界 clamp（防止越界）
        y = max(1.0, min(y, H - 2.0))
        x = max(1.0, min(x, W - 2.0))
        
        # 保存原位置，用于最后计算实际 velocity
        old_y, old_x = y, x
        
        yi = int(y)
        xi = int(x)
        dist = sdf[yi, xi]  # 有符号：正=可行区内，负=可行区外
        is_on_road = dist > 0
        
        # === 1. 导航力 ===
        nav_dir = bilinear_sample(field, y, x)
        nav_vy = nav_dir[0] * v0
        nav_vx = nav_dir[1] * v0
        
        # === 2. 墙壁斥力 / 掉网恢复 ===
        wall_vy, wall_vx = 0.0, 0.0
        
        # 计算 SDF 梯度（指向 SDF 增大的方向，即指向可行区内部）
        grad_y = (sdf[min(yi+1, H-1), xi] - sdf[max(yi-1, 0), xi]) * 0.5
        grad_x = (sdf[yi, min(xi+1, W-1)] - sdf[yi, max(xi-1, 0)]) * 0.5
        grad_mag = np.sqrt(grad_y * grad_y + grad_x * grad_x) + 1e-6
        
        if not is_on_road:
            # 掉网：只用 SDF 梯度回到可行区，忽略导航力
            wall_vy = (grad_y / grad_mag) * off_road_recovery
            wall_vx = (grad_x / grad_mag) * off_road_recovery
            nav_vy = 0.0
            nav_vx = 0.0
            
        elif dist < wall_dist_thresh:
            # 靠近边界：柔性斥力，推向可行区内部
            push = wall_push_strength * (1.0 - dist / wall_dist_thresh)
            wall_vy = (grad_y / grad_mag) * push
            wall_vx = (grad_x / grad_mag) * push
        
        # === 3. 噪声 ===
        ny = noise_sigma * np.random.randn()
        nx = noise_sigma * np.random.randn()
        
        # === 4. 目标速度 ===
        target_vy = nav_vy + wall_vy + ny
        target_vx = nav_vx + wall_vx + nx
        
        # === 5. 动量混合 ===
        # 新速度 = momentum * 旧速度 + (1-momentum) * 目标速度
        eff_vy = momentum * prev_vy + (1.0 - momentum) * target_vy
        eff_vx = momentum * prev_vx + (1.0 - momentum) * target_vx
        
        # 速度限制
        speed = np.sqrt(eff_vy * eff_vy + eff_vx * eff_vx)
        max_speed = v0 * 2.0
        if speed > max_speed:
            eff_vy = eff_vy / speed * max_speed
            eff_vx = eff_vx / speed * max_speed
        
        # 确保最小速度（防止停滞）
        if speed < 0.1 and is_on_road:
            # 如果速度太低但在道路上，直接使用导航方向
            eff_vy = nav_vy
            eff_vx = nav_vx
        
        # 位置更新
        new_y = y + eff_vy * dt
        new_x = x + eff_vx * dt
        
        # 边界约束
        new_y = max(1.0, min(new_y, H - 2.0))
        new_x = max(1.0, min(new_x, W - 2.0))
        
        # === 6. 道路约束：如果新位置脱离道路，尝试沿道路方向步进 ===
        new_yi = int(new_y)
        new_xi = int(new_x)
        if sdf[new_yi, new_xi] < 0:
            best_y, best_x = y, x
            best_dot = -2.0
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                cny = yi + dy
                cnx = xi + dx
                if 0 <= cny < H and 0 <= cnx < W:
                    if sdf[cny, cnx] > 0:
                        dot = dy * eff_vy + dx * eff_vx
                        if dot > best_dot:
                            best_dot = dot
                            best_y = float(cny)
                            best_x = float(cnx)
            
            new_y = best_y
            new_x = best_x
        
        pos[i, 0] = new_y
        pos[i, 1] = new_x
        
        # 计算实际 velocity = (new_pos - old_pos) / dt
        # 这确保 velocity 反映真实移动，而非动量混合后的「目标速度」
        vel[i, 0] = (new_y - old_y) / dt
        vel[i, 1] = (new_x - old_x) / dt
    
    return 0
