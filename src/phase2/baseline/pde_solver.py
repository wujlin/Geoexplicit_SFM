"""
有限差分扩散求解（Track A Baseline）。
- 在 walkable_mask 上做平滑扩散，非可走区域扩散系数降低（近似绝缘）。
- 返回平滑后的场以及梯度/score 场，供流线或导航使用。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

LAPLACE_KERNEL = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)


def diffuse_density(
    target_density: np.ndarray,
    walkable_mask: np.ndarray,
    num_iters: int = 400,
    alpha: float = 0.15,
    base_diffusivity: float = 1e-3,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    通过离散拉普拉斯迭代平滑目标密度。
    - walkable_mask 为 0/1；非可走区域扩散系数接近 0。
    - alpha 越小越稳定；建议 alpha <= 0.25。
    """
    field = np.asarray(target_density, dtype=float).copy()
    mask = (np.asarray(walkable_mask) > 0).astype(float)
    diffusivity = base_diffusivity + (1.0 - base_diffusivity) * mask

    for _ in range(num_iters):
        lap = convolve(field, LAPLACE_KERNEL, mode="nearest")
        field = field + alpha * diffusivity * lap
        if clamp_min is not None:
            field[field < clamp_min] = clamp_min
        if clamp_max is not None:
            field[field > clamp_max] = clamp_max

    if normalize:
        field = field / (field.max() + 1e-9)
    return field


def compute_score_field(field: np.ndarray, eps: float = 1e-6, normalize: bool = True):
    """
    计算梯度与 score（梯度/场强）。
    返回 (grad_y, grad_x), (score_y, score_x)
    """
    grad_y, grad_x = np.gradient(field)
    score_y = grad_y / (field + eps)
    score_x = grad_x / (field + eps)
    if normalize:
        mag = np.hypot(score_y, score_x) + 1e-9
        score_y = score_y / mag
        score_x = score_x / mag
    return (grad_y, grad_x), (score_y, score_x)


def solve_field(
    walkable_mask: np.ndarray,
    target_density: np.ndarray,
    num_iters: int = 400,
    alpha: float = 0.15,
    base_diffusivity: float = 1e-3,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = None,
    normalize: bool = True,
):
    """
    便捷封装：扩散 -> 梯度 -> score。
    返回 dict，包括 smooth_field、grad、score。
    """
    smooth = diffuse_density(
        target_density,
        walkable_mask,
        num_iters=num_iters,
        alpha=alpha,
        base_diffusivity=base_diffusivity,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        normalize=normalize,
    )
    grad, score = compute_score_field(smooth, normalize=True)
    return {"smooth_field": smooth, "grad": grad, "score": score}
