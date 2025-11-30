from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, parallel=True)
def bilinear_sample(field, y, x):
    """
    field: (2, H, W)
    y, x: scalar
    返回 (2,) 双线性插值
    """
    h, w = field.shape[1], field.shape[2]
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
    dt,
    tau,  # 过阻尼模式下未使用
    noise_sigma,
    v0,
    respawn_radius,
    pixel_scale,  # 过阻尼模式下未使用
):
    """
    过阻尼 Langevin：直接用场作为速度，加入噪声，靠近边界时用 SDF 梯度推回。
    """
    N = pos.shape[0]
    H, W = sdf.shape
    PUSH_BACK = 1.0
    for i in numba.prange(N):
        if not active[i]:
            continue
        y, x = pos[i, 0], pos[i, 1]
        vy, vx = bilinear_sample(field, y, x)
        mag = np.sqrt(vy * vy + vx * vx) + 1e-9
        if mag > 1e-3:
            vy = (vy / mag) * v0
            vx = (vx / mag) * v0
        else:
            vy, vx = 0.0, 0.0

        yi = int(y)
        xi = int(x)
        yi = 1 if yi < 1 else (H - 2 if yi > H - 2 else yi)
        xi = 1 if xi < 1 else (W - 2 if xi > W - 2 else xi)
        dist = sdf[yi, xi]
        wall_vy, wall_vx = 0.0, 0.0
        if dist < 1.0:
            grad_y = (sdf[yi + 1, xi] - sdf[yi - 1, xi]) * 0.5
            grad_x = (sdf[yi, xi + 1] - sdf[yi, xi - 1]) * 0.5
            push = (1.0 - dist) * PUSH_BACK / dt
            wall_vy = grad_y * push
            wall_vx = grad_x * push

        ny = noise_sigma * np.random.randn()
        nx = noise_sigma * np.random.randn()

        eff_vy = vy + wall_vy + ny
        eff_vx = vx + wall_vx + nx

        pos[i, 0] += eff_vy * dt
        pos[i, 1] += eff_vx * dt

        vel[i, 0] = eff_vy
        vel[i, 1] = eff_vx

        if pos[i, 0] < 0.1:
            pos[i, 0] = 0.1
        if pos[i, 0] > H - 1.1:
            pos[i, 0] = H - 1.1
        if pos[i, 1] < 0.1:
            pos[i, 1] = 0.1
        if pos[i, 1] > W - 1.1:
            pos[i, 1] = W - 1.1

        if respawn_radius > 0.0 and dist <= respawn_radius:
            active[i] = False
    return 0
