import json
from pathlib import Path

import numpy as np
import torch

from src.phase2 import config
from src.phase2.common.visualizer import plot_score_samples
from src.phase2.innovation.network import UNetSmall


def load_model(device: torch.device, model_path: Path = config.INNOVATION_MODEL_PATH):
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


def infer_full_field(model: torch.nn.Module, mask: np.ndarray, density: np.ndarray, sigma: float, device):
    h, w = density.shape
    y_grid = np.linspace(-1, 1, h, dtype=np.float32)
    x_grid = np.linspace(-1, 1, w, dtype=np.float32)
    yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")
    inp = np.stack([mask.astype(np.float32), density.astype(np.float32), yy, xx], axis=0)
    with torch.no_grad():
        pred_noise = model(torch.from_numpy(inp[None]).to(device))  # (1,2,H,W) 噪声
    pred_noise = pred_noise.squeeze(0).cpu().numpy()
    field_score = -pred_noise / sigma  # 转为 score
    return field_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    density = np.load(config.TARGET_DENSITY_PATH)
    mask = np.load(config.WALKABLE_MASK_PATH)
    model, sigma = load_model(device)

    field = infer_full_field(model, mask, density, sigma=sigma, device=device)  # (2, H, W) score
    np.save(config.FIELD_INNOVATION_PATH, field)
    print(f"保存神经场(score): {config.FIELD_INNOVATION_PATH}")

    # 稀疏采样用于可视化
    stride = 20
    h, w = density.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    coords = np.array([[y, x] for y in ys for x in xs], dtype=float)
    vecs = np.stack(
        [
            field[0, coords[:, 0].astype(int), coords[:, 1].astype(int)],
            field[1, coords[:, 0].astype(int), coords[:, 1].astype(int)],
        ],
        axis=-1,
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
