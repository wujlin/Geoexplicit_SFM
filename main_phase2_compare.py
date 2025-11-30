import json
from pathlib import Path

import numpy as np
import torch

from src.phase2 import config
from src.phase2.baseline import solve_field
from src.phase2.common.comparison import plot_comparison, compute_vector_stats
from src.phase2.innovation.network import UNetSmall


def infer_full_field(model, mask, density, device="cpu"):
    inp = np.stack([mask.astype(np.float32), density.astype(np.float32)], axis=0)
    with torch.no_grad():
        pred = model(torch.from_numpy(inp[None]).to(device))
    field = pred.squeeze(0).cpu().numpy()  # (2, H, W)
    return field


def main():
    mask = np.load(config.WALKABLE_MASK_PATH)
    density = np.load(config.TARGET_DENSITY_PATH)

    # baseline score 使用 PDE 结果
    baseline = solve_field(mask, density, num_iters=400, alpha=0.15, base_diffusivity=1e-3, clamp_min=0.0, normalize=True)
    baseline_score = np.stack(baseline["score"], axis=-1)  # (H, W, 2)
    # 稀疏抽样 baseline
    stride = 20
    h, w = density.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    coords_base = np.array([[y, x] for y in ys for x in xs], dtype=float)
    vecs_base = baseline_score[coords_base[:, 0].astype(int), coords_base[:, 1].astype(int)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Path(config.FIELD_INNOVATION_PATH).exists():
        field_innov = np.load(config.FIELD_INNOVATION_PATH)  # (2, H, W)
    else:
        model = UNetSmall(in_channels=2, base_channels=32).to(device)
        state = torch.load(config.INNOVATION_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        field_innov = infer_full_field(model, mask, density, device=device)

    coords_innov = coords_base
    vecs_innov = np.stack(
        [
            field_innov[0, coords_innov[:, 0].astype(int), coords_innov[:, 1].astype(int)],
            field_innov[1, coords_innov[:, 0].astype(int), coords_innov[:, 1].astype(int)],
        ],
        axis=-1,
    )

    out_img = plot_comparison(
        background=density,
        mask=mask,
        coords_a=coords_base,
        vecs_a=vecs_base,
        coords_b=coords_innov,
        vecs_b=vecs_innov,
        labels=("Baseline", "Innovation"),
        out_path=config.COMPARISON_VIZ_PATH,
        scale=30.0,
    )
    print(f"保存对比图: {out_img}")

    stats = compute_vector_stats(vecs_base, vecs_innov)
    stats_path = Path(str(config.COMPARISON_VIZ_PATH) + ".json")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"保存对比统计: {stats_path}")


if __name__ == "__main__":
    main()
