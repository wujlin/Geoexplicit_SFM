"""
Phase 3 物理模块 - 个体目的地版本

每个 agent 使用各自目的地的导航场进行导航。
不使用 Numba 并行化（因为需要从 NavFieldManager 动态获取导航场）。
"""

from __future__ import annotations

import numpy as np


def step_kernel_individual_dest(
    pos: np.ndarray,
    vel: np.ndarray,
    active: np.ndarray,
    dest: np.ndarray,
    arrived: np.ndarray,
    sdf: np.ndarray,
    mask: np.ndarray,
    nav_field_manager,
    dt: float,
    noise_sigma: float,
    v0: float,
    wall_dist_thresh: float,
    wall_push_strength: float,
    off_road_recovery: float,
    momentum: float,
    arrival_threshold: float,
) -> list:
    """
    个体目的地版本的物理更新内核
    
    与原版差异：
    1. 每个 agent 使用各自 dest 对应的导航场
    2. 检测到达条件，返回到达的 agent 索引
    
    Args:
        pos: (N, 2) 位置 [y, x]
        vel: (N, 2) 速度 [vy, vx]
        active: (N,) 活跃标志
        dest: (N,) 目的地 sink ID
        arrived: (N,) 到达标志（用于统计）
        sdf: (H, W) 有符号距离场
        mask: (H, W) 可行走掩膜
        nav_field_manager: 导航场管理器
        arrival_threshold: 到达阈值（像素）
        ...其他物理参数
        
    Returns:
        arrived_indices: 本步到达的 agent 索引列表
    """
    N = pos.shape[0]
    H, W = sdf.shape
    
    arrived_indices = []
    
    for i in range(N):
        if not active[i]:
            continue
        
        y, x = pos[i, 0], pos[i, 1]
        prev_vy, prev_vx = vel[i, 0], vel[i, 1]
        dest_sink = int(dest[i])
        
        # 边界 clamp
        y = max(1.0, min(y, H - 2.0))
        x = max(1.0, min(x, W - 2.0))
        
        old_y, old_x = y, x
        
        yi = int(y)
        xi = int(x)
        dist = sdf[yi, xi]
        is_on_road = dist > 0
        
        # === 检测到达 ===
        # 使用导航场中的距离场检测到达
        dist_to_dest = nav_field_manager.get_distance(dest_sink, yi, xi)
        if dist_to_dest < arrival_threshold:
            arrived_indices.append(i)
            continue  # 跳过更新，让 engine 处理重生
        
        # === 1. 获取导航方向 ===
        nav_y, nav_x = nav_field_manager.get_nav_direction(dest_sink, yi, xi)
        nav_vy = nav_y * v0
        nav_vx = nav_x * v0
        
        # === 2. 墙壁斥力 / 掉网恢复 ===
        wall_vy, wall_vx = 0.0, 0.0
        
        # 计算 SDF 梯度
        grad_y = (sdf[min(yi+1, H-1), xi] - sdf[max(yi-1, 0), xi]) * 0.5
        grad_x = (sdf[yi, min(xi+1, W-1)] - sdf[yi, max(xi-1, 0)]) * 0.5
        grad_mag = np.sqrt(grad_y**2 + grad_x**2) + 1e-6
        
        if not is_on_road:
            # 掉网：只用 SDF 梯度回到可行区
            wall_vy = (grad_y / grad_mag) * off_road_recovery
            wall_vx = (grad_x / grad_mag) * off_road_recovery
            nav_vy = 0.0
            nav_vx = 0.0
        elif dist < wall_dist_thresh:
            # 靠近边界：柔性斥力
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
        eff_vy = momentum * prev_vy + (1.0 - momentum) * target_vy
        eff_vx = momentum * prev_vx + (1.0 - momentum) * target_vx
        
        # 速度限制
        speed = np.sqrt(eff_vy**2 + eff_vx**2)
        max_speed = v0 * 2.0
        if speed > max_speed:
            eff_vy = eff_vy / speed * max_speed
            eff_vx = eff_vx / speed * max_speed
        
        # 确保最小速度
        if speed < 0.1 and is_on_road:
            eff_vy = nav_vy
            eff_vx = nav_vx
        
        # 位置更新
        new_y = y + eff_vy * dt
        new_x = x + eff_vx * dt
        
        # 边界约束
        new_y = max(1.0, min(new_y, H - 2.0))
        new_x = max(1.0, min(new_x, W - 2.0))
        
        # === 6. 道路约束 ===
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
        
        # 计算实际 velocity
        vel[i, 0] = (new_y - old_y) / dt
        vel[i, 1] = (new_x - old_x) / dt
    
    return arrived_indices
