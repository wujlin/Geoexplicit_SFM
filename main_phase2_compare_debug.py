"""
比较 Baseline vs Innovation，加入：
- 预测噪声转换为 score（-eps/sigma）
- XY 通道交换测试
- 只在可走区域计算相似度
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.phase2 import config
from src.phase2.baseline import solve_field
from src.phase2.common.comparison import compute_vector_stats, plot_comparison
from src.phase2.innovation.network import UNetSmall
from src.phase2.innovation.trainer import _sample_pred_at_coords


def _load_model(device: torch.device, model_path: Path = config.INNOVATION_MODEL_PATH):
    model_path = Path(model_path)
    base_channels = 32
    sigma = 10.0
    meta_path = Path(str(model_path) + ".json")
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            base_channels = int(meta.get("base_channels", base_channels))
            sigma = float(meta.get("sigma", sigma))
        except Exception:
            pass
    model = UNetSmall(in_channels=4, base_channels=base_channels).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, sigma


def _infer_noise_at_coords(model, mask, density, coords_np, device):
    """在给定坐标上采样模型输出的噪声（未缩放）。"""
    h, w = density.shape
    y_grid = np.linspace(-1, 1, h, dtype=np.float32)
    x_grid = np.linspace(-1, 1, w, dtype=np.float32)
    yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")
    inp = np.stack([mask, density, yy, xx], axis=0).astype(np.float32)
    inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
    coords_t = torch.from_numpy(coords_np).to(device)
    with torch.no_grad():
        pred_noise_map = model(inp_t)  # (1,2,H,W)
        pred_noise_vecs = _sample_pred_at_coords(pred_noise_map, coords_t)  # (N,2)
    return pred_noise_vecs.cpu().numpy()


def main():
    mask = np.load(config.WALKABLE_MASK_PATH)
    density = np.load(config.TARGET_DENSITY_PATH)

    # Baseline
    baseline = solve_field(mask, density, num_iters=400, alpha=0.15, base_diffusivity=1e-3, clamp_min=0.0, normalize=True)
    score_y, score_x = baseline["score"]

    stride = 25
    h, w = density.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([YY.ravel(), XX.ravel()], axis=1).astype(np.float32)

    vecs_base = np.stack(
        [
            score_y[YY, XX],
            score_x[YY, XX],
        ],
        axis=-1,
    ).reshape(-1, 2)

    # Innovation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sigma = _load_model(device)
    pred_noise_vecs = _infer_noise_at_coords(model, mask, density, coords, device=device)
    vecs_innov = -1.0 * pred_noise_vecs / sigma  # 转换为 score

    # Case A: 原始
    stats_a = compute_vector_stats(vecs_base, vecs_innov)
    # Case B: 交换通道
    vecs_innov_flip = vecs_innov[:, [1, 0]]
    stats_b = compute_vector_stats(vecs_base, vecs_innov_flip)

    better_is_flip = stats_b["cos_mean"] > stats_a["cos_mean"]
    final_vecs = vecs_innov_flip if better_is_flip else vecs_innov
    label = "Innovation (Swapped)" if better_is_flip else "Innovation"
    stats_final = stats_b if better_is_flip else stats_a

    # 只在路网上统计
    on_road = mask[coords[:, 0].astype(int), coords[:, 1].astype(int)] > 0.5
    if on_road.any():
        stats_road = compute_vector_stats(vecs_base[on_road], final_vecs[on_road])
    else:
        stats_road = None

    # 输出统计
    print(f"Case A (orig) cos_mean={stats_a['cos_mean']:.4f}")
    print(f"Case B (swap) cos_mean={stats_b['cos_mean']:.4f}")
    if stats_road:
        print(f"On-road cos_mean={stats_road['cos_mean']:.4f}, median={stats_road['cos_median']:.4f}")

    # 绘图
    out_img = plot_comparison(
        background=density,
        mask=mask,
        coords_a=coords,
        vecs_a=vecs_base,
        coords_b=coords,
        vecs_b=final_vecs,
        labels=("Baseline", label),
        out_path=config.COMPARISON_VIZ_PATH,
        scale=30.0,
    )
    print(f"Saved: {out_img}")


if __name__ == "__main__":
    main()
