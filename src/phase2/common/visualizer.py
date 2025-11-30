"""
Baseline 场可视化：热力图 + 可选 score 箭头。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_field(
    field: np.ndarray,
    mask: np.ndarray,
    score: tuple[np.ndarray, np.ndarray] | None = None,
    out_path: Path | str = "baseline_field.png",
    title: str = "Baseline Field",
    max_quiver: int = 40,
):
    """
    简易可视化：非可走区域置灰，场值用 colormap；score 用稀疏箭头。
    """
    field = np.asarray(field)
    mask = np.asarray(mask)
    h, w = field.shape

    fig, ax = plt.subplots(figsize=(12, 8))
    bg = np.ma.masked_where(mask > 0, field)
    ax.imshow(np.where(mask > 0, 0.0, 1.0), cmap="gray", origin="lower", alpha=0.2)
    im = ax.imshow(field, cmap="inferno", origin="lower", alpha=0.9)
    ax.imshow(bg, cmap="gray_r", origin="lower", alpha=0.6)

    if score is not None:
        sy, sx = score
        sy = np.asarray(sy)
        sx = np.asarray(sx)
        step_y = max(h // max_quiver, 1)
        step_x = max(w // max_quiver, 1)
        y_idx = np.arange(0, h, step_y)
        x_idx = np.arange(0, w, step_x)
        X, Y = np.meshgrid(x_idx, y_idx)
        ax.quiver(
            X,
            Y,
            sx[::step_y, ::step_x],
            sy[::step_y, ::step_x],
            color="cyan",
            alpha=0.6,
            scale=50,
            width=0.0025,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="field value")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_score_samples(
    background: np.ndarray,
    mask: np.ndarray,
    coords: np.ndarray,
    vecs: np.ndarray,
    out_path: Path | str = "score_field.png",
    title: str = "Score Field (samples)",
    scale: float = 30.0,
):
    """
    使用稀疏采样的向量进行可视化，背景可用 density 或 baseline 场。
    """
    bg = np.asarray(background)
    mask = np.asarray(mask)
    coords = np.asarray(coords)
    vecs = np.asarray(vecs)

    fig, ax = plt.subplots(figsize=(12, 8))
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


if __name__ == "__main__":
    # 便于快速手动运行
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--field", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--score", default=None)
    parser.add_argument("--out", default="baseline_field.png")
    args = parser.parse_args()

    field = np.load(args.field)
    mask = np.load(args.mask)
    score = None
    if args.score:
        npz = np.load(args.score)
        score = (npz["score_y"], npz["score_x"])
    plot_field(field, mask, score=score, out_path=args.out)
