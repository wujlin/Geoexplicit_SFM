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
):
    """
    pos, vel: (N,2)
    active: (N,)
    field: (2,H,W) - score/导航场
    sdf: (H,W) - distance transform
    返回 escaped_indices 列表
    """
    N = pos.shape[0]
    escaped = []
    for i in numba.prange(N):
        if not active[i]:
            continue
        y, x = pos[i, 0], pos[i, 1]
        fy, fx = bilinear_sample(field, y, x)
        # 归一化导航力
        normf = np.sqrt(fy * fy + fx * fx) + 1e-6
        fy /= normf
        fx /= normf

        # 噪声
        ny = noise_sigma * np.random.randn()
        nx = noise_sigma * np.random.randn()

        # 简单墙壁斥力：基于 sdf 梯度符号（粗略）
        sy = sdf[int(y), int(x)]
        # 速度更新
        vel[i, 0] += ((fy * v0 - vel[i, 0]) / tau) * dt + ny * np.sqrt(dt)
        vel[i, 1] += ((fx * v0 - vel[i, 1]) / tau) * dt + nx * np.sqrt(dt)

        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt

        # 边界裁剪
        if pos[i, 0] < 0:
            pos[i, 0] = 0
        if pos[i, 0] > field.shape[1] - 1:
            pos[i, 0] = field.shape[1] - 1
        if pos[i, 1] < 0:
            pos[i, 1] = 0
        if pos[i, 1] > field.shape[2] - 1:
            pos[i, 1] = field.shape[2] - 1

        # 到达判定：当前 sdf 小于阈值
        if sy <= respawn_radius:
            escaped.append(i)
            active[i] = False
    return escaped
