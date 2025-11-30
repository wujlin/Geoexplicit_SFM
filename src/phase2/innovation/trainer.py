"""
Score Field 训练（Denoising Score Matching）：
- 从 target_density 按概率采样点，加入高斯噪声
- 构造 query heatmap，网络输出 score 场
- 在噪声点位置处采样预测向量，与理论 score 目标做 MSE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import sys
from pathlib import Path

# 确保项目根目录在 sys.path，支持直接 python trainer.py 运行
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.phase2 import config
from src.phase2.innovation.network import UNetSmall

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - 可选依赖
    tqdm = None


def _sample_indices_from_density(density: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    flat = density.reshape(-1)
    flat = flat + 1e-9
    probs = flat / flat.sum()
    idx = rng.choice(len(flat), p=probs)
    h, w = density.shape
    y = idx // w
    x = idx % w
    return int(y), int(x)


def _gaussian_query(h: int, w: int, y: float, x: float, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2))


class ScoreDataset(Dataset):
    def __init__(
        self,
        density: np.ndarray,
        mask: np.ndarray,
        sigma: float = 4.0,
        query_sigma: float = 2.0,
        n_samples: int = 50000,
        seed: int = 42,
    ):
        self.density = density.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.sigma = sigma
        self.query_sigma = query_sigma
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.h, self.w = density.shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        y0, x0 = _sample_indices_from_density(self.density, self.rng)
        noise = self.rng.normal(loc=0.0, scale=self.sigma, size=2)
        y_noisy = float(np.clip(y0 + noise[0], 0, self.h - 1))
        x_noisy = float(np.clip(x0 + noise[1], 0, self.w - 1))

        query = _gaussian_query(self.h, self.w, y_noisy, x_noisy, sigma=self.query_sigma).astype(np.float32)
        inp = np.stack([self.mask, self.density, query], axis=0)

        target_vec = np.array([(y0 - y_noisy) / (self.sigma**2), (x0 - x_noisy) / (self.sigma**2)], dtype=np.float32)
        coord = np.array([y_noisy, x_noisy], dtype=np.float32)
        return torch.from_numpy(inp), torch.from_numpy(target_vec), torch.from_numpy(coord)


def _sample_pred_at_coords(pred: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    pred: (B, 2, H, W)
    coords: (B, 2) in (y, x)
    返回 (B, 2) 在坐标处的双线性采样值。
    """
    b, _, h, w = pred.shape
    y = coords[:, 0]
    x = coords[:, 1]
    x_norm = 2 * (x / (w - 1)) - 1
    y_norm = 2 * (y / (h - 1)) - 1
    grid = torch.stack([x_norm, y_norm], dim=1).view(b, 1, 1, 2)
    sampled = F.grid_sample(pred, grid, align_corners=True)
    return sampled.view(b, 2)


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-3
    num_steps: int = 2000
    sigma: float = 4.0
    query_sigma: float = 2.0
    log_interval: int = 100
    device: str = "cpu"


def train_dsm(
    density_path: str | None = None,
    mask_path: str | None = None,
    model_out: str | None = None,
    cfg: TrainConfig | None = None,
):
    cfg = cfg or TrainConfig()
    density_np = np.load(density_path or config.TARGET_DENSITY_PATH)
    mask_np = np.load(mask_path or config.WALKABLE_MASK_PATH)

    dataset = ScoreDataset(
        density_np,
        mask_np,
        sigma=cfg.sigma,
        query_sigma=cfg.query_sigma,
        n_samples=cfg.num_steps * cfg.batch_size,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)

    device = torch.device(cfg.device)
    model = UNetSmall(in_channels=3, base_channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=min(cfg.num_steps, len(loader)), ncols=80, desc="train")

    global_step = 0
    for batch in iterator:
        opt.zero_grad()
        inp, target_vec, coords = batch
        inp = inp.to(device)
        target_vec = target_vec.to(device)
        coords = coords.to(device)

        pred_field = model(inp)
        pred_vec = _sample_pred_at_coords(pred_field, coords)
        loss = ((pred_vec - target_vec) ** 2).sum(dim=1).mean()
        loss.backward()
        opt.step()

        if global_step % cfg.log_interval == 0:
            print(f"[step {global_step}] loss={loss.item():.6f}")
        if tqdm is not None:
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        global_step += 1
        if global_step >= cfg.num_steps:
            break

    out_path = model_out or config.INNOVATION_MODEL_PATH
    torch.save(model.state_dict(), out_path)
    print(f"模型已保存: {out_path}")
    return model


if __name__ == "__main__":
    train_dsm()
