import numpy as np
import torch

from src.phase2 import config
from src.phase2.common.visualizer import plot_score_samples
from src.phase2.innovation.network import UNetSmall


def load_model(device: torch.device):
    model = UNetSmall(in_channels=2, base_channels=32).to(device)
    state = torch.load(config.INNOVATION_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_full_field(model: torch.nn.Module, mask: np.ndarray, density: np.ndarray, device: torch.device | str = "cpu"):
    inp = np.stack([mask.astype(np.float32), density.astype(np.float32)], axis=0)
    with torch.no_grad():
        pred = model(torch.from_numpy(inp[None]).to(device))
    field = pred.squeeze(0).cpu().numpy()  # (2, H, W)
    return field


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    density = np.load(config.TARGET_DENSITY_PATH)
    mask = np.load(config.WALKABLE_MASK_PATH)
    model = load_model(device)

    field = infer_full_field(model, mask, density, device=device)  # (2, H, W)
    np.save(config.FIELD_INNOVATION_PATH, field)
    print(f"保存神经场: {config.FIELD_INNOVATION_PATH}")

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
