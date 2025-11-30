"""
并排对比 Baseline 与 Innovation 的向量场（稀疏采样）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(
    background: np.ndarray,
    mask: np.ndarray,
    coords_a: np.ndarray,
    vecs_a: np.ndarray,
    coords_b: np.ndarray,
    vecs_b: np.ndarray,
    labels=("Baseline", "Innovation"),
    out_path: Path | str = "comparison.png",
    scale: float = 30.0,
):
    bg = np.asarray(background)
    mask = np.asarray(mask)
    coords_a = np.asarray(coords_a)
    vecs_a = np.asarray(vecs_a)
    coords_b = np.asarray(coords_b)
    vecs_b = np.asarray(vecs_b)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    for ax, coords, vecs, title in zip(axes, (coords_a, coords_b), (vecs_a, vecs_b), labels):
        ax.imshow(bg, cmap="inferno", origin="lower", alpha=0.85)
        ax.imshow(np.where(mask > 0, 0.0, 1.0), cmap="gray", origin="lower", alpha=0.2)
        ax.quiver(
            coords[:, 1],
            coords[:, 0],
            vecs[:, 1],
            vecs[:, 0],
            color="cyan",
            alpha=0.8,
            scale=scale,
            width=0.003,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def compute_vector_stats(vecs_a: np.ndarray, vecs_b: np.ndarray) -> dict:
    """
    计算方向相似度、角度误差、差分等统计。
    """
    va = np.asarray(vecs_a)
    vb = np.asarray(vecs_b)
    norm_a = np.linalg.norm(va, axis=1) + 1e-9
    norm_b = np.linalg.norm(vb, axis=1) + 1e-9
    cos = (va * vb).sum(axis=1) / (norm_a * norm_b)
    cos = np.clip(cos, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos))
    diff = np.linalg.norm(va - vb, axis=1)
    stats = {
        "cos_mean": float(cos.mean()),
        "cos_median": float(np.median(cos)),
        "cos_p10": float(np.percentile(cos, 10)),
        "cos_p90": float(np.percentile(cos, 90)),
        "angle_mean_deg": float(angle_deg.mean()),
        "angle_median_deg": float(np.median(angle_deg)),
        "angle_p90_deg": float(np.percentile(angle_deg, 90)),
        "diff_mean": float(diff.mean()),
        "diff_median": float(np.median(diff)),
        "norm_a_mean": float(norm_a.mean()),
        "norm_b_mean": float(norm_b.mean()),
    }
    return stats
