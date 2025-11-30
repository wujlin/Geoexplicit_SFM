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
    tau,
    noise_sigma,
    v0,
    respawn_radius,
    pixel_scale,
):
    """
    pos, vel: (N,2)
    active: (N,)
    field: (2,H,W) - score/导航场
    sdf: (H,W) - distance transform，正值在可行走区域
    墙壁斥力：基于 sdf 梯度，避免掉出路网
    """
    N = pos.shape[0]
    H, W = sdf.shape
    WALL_STIFFNESS = 20.0
    for i in numba.prange(N):
        if not active[i]:
            continue
        y, x = pos[i, 0], pos[i, 1]
        fy, fx = bilinear_sample(field, y, x)
        mag = np.sqrt(fy * fy + fx * fx) + 1e-9
        if mag > 1e-3:
            fy /= mag
            fx /= mag

        # 墙壁斥力：SDF 梯度指向路心
        yi = int(y)
        xi = int(x)
        yi = 1 if yi < 1 else (H - 2 if yi > H - 2 else yi)
        xi = 1 if xi < 1 else (W - 2 if xi > W - 2 else xi)
        grad_y = (sdf[yi + 1, xi] - sdf[yi - 1, xi]) * 0.5
        grad_x = (sdf[yi, xi + 1] - sdf[yi, xi - 1]) * 0.5
        dist = sdf[yi, xi]
        f_wall_y = 0.0
        f_wall_x = 0.0
        if dist < 3.0:
            strength = WALL_STIFFNESS * (3.0 - dist)
            f_wall_y = grad_y * strength
            f_wall_x = grad_x * strength

        ny = noise_sigma * np.random.randn()
        nx = noise_sigma * np.random.randn()

        vel[i, 0] += ((fy * v0 - vel[i, 0]) / tau + f_wall_y) * dt + ny * np.sqrt(dt)
        vel[i, 1] += ((fx * v0 - vel[i, 1]) / tau + f_wall_x) * dt + nx * np.sqrt(dt)

        pos[i, 0] += (vel[i, 0] * dt) / pixel_scale
        pos[i, 1] += (vel[i, 1] * dt) / pixel_scale

        # 边界裁剪
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
