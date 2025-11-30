import math
from pathlib import Path

import numpy as np
import torch

from src.phase2 import config
from src.phase2.common.visualizer import plot_score_samples
from src.phase2.innovation.network import UNetSmall
from src.phase2.innovation.trainer import _gaussian_query, _sample_pred_at_coords


def load_model(device: torch.device):
    model = UNetSmall(in_channels=3, base_channels=32).to(device)
    state = torch.load(config.INNOVATION_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_sparse_field(
    model: torch.nn.Module,
    mask: np.ndarray,
    density: np.ndarray,
    stride: int = 20,
    query_sigma: float = 1.5,
    device: torch.device | str = "cpu",
):
    h, w = density.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    coords = []
    vecs = []
    with torch.no_grad():
        for y in ys:
            for x in xs:
                query = _gaussian_query(h, w, y, x, sigma=query_sigma).astype(np.float32)
                inp = np.stack([mask.astype(np.float32), density.astype(np.float32), query], axis=0)
                inp_t = torch.from_numpy(inp[None]).to(device)
                pred_field = model(inp_t)
                pred_vec = _sample_pred_at_coords(pred_field, torch.tensor([[y, x]], device=device, dtype=torch.float32))
                vec = pred_vec.squeeze(0).cpu().numpy()
                coords.append([y, x])
                vecs.append(vec)
    return np.array(coords, dtype=float), np.array(vecs, dtype=float)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    density = np.load(config.TARGET_DENSITY_PATH)
    mask = np.load(config.WALKABLE_MASK_PATH)
    model = load_model(device)

    coords, vecs = infer_sparse_field(
        model,
        mask=mask,
        density=density,
        stride=20,
        query_sigma=1.5,
        device=device,
    )
    np.savez(config.INNOVATION_SAMPLES_PATH, coords=coords, vecs=vecs)
    print(f"保存采样向量: {config.INNOVATION_SAMPLES_PATH}, n={len(coords)}")

    out_img = plot_score_samples(
        background=density,
        mask=mask,
        coords=coords,
        vecs=vecs,
        out_path=config.INNOVATION_VIZ_PATH,
        title="Innovation Score Field (sparse)",
        scale=30.0,
    )
    print(f"保存可视化: {out_img}")


if __name__ == "__main__":
    main()
