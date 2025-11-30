"""
Score Field 训练（One-Map DSM 优化版）：
- 静态输入（mask+density）只前向一次；在同一张图上随机采样大量坐标，直接预测噪声方向。
- 目标为标准高斯噪声（带负号代表恢复方向），避免数值过小导致模型“躺平”。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
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


def _sample_indices_from_density(density: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    flat = density.reshape(-1)
    flat = flat + 1e-9
    probs = flat / flat.sum()
    idx = rng.choice(len(flat), size=n, p=probs)
    h, w = density.shape
    y = idx // w
    x = idx % w
    return y, x


class ScoreDataset(Dataset):
    """
    轻量 Dataset：预生成采样点与噪声目标，不返回图像（图像固定在训练循环中）。
    """

    def __init__(
        self,
        density: np.ndarray,
        mask: np.ndarray,
        sigma: float = 10.0,
        n_samples: int = 50000,
        seed: int = 42,
    ):
        self.density = density.astype(np.float32)
        self.sigma = sigma
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.h, self.w = density.shape

        # 预生成采样与噪声，提升性能
        print(f"Pre-generating {n_samples} samples with sigma={sigma} ...")
        y0s, x0s = _sample_indices_from_density(self.density, self.rng, n_samples)
        eps = self.rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2)).astype(np.float32)
        y_noisy = np.clip(y0s + eps[:, 0] * sigma, 0, self.h - 1)
        x_noisy = np.clip(x0s + eps[:, 1] * sigma, 0, self.w - 1)
        self.coords = np.stack([y_noisy, x_noisy], axis=1).astype(np.float32)  # (N,2)
        self.targets = -1.0 * eps  # 恢复方向

    def __len__(self):
        return self.n_samples

    def reset_rng(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        return torch.from_numpy(self.targets[idx]), torch.from_numpy(self.coords[idx])


def _sample_pred_at_coords(pred: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    pred: (1, 2, H, W) 静态大图
    coords: (B, 2) in (y, x)
    返回 (B, 2) 在坐标处的双线性采样值。
    """
    _, _, h, w = pred.shape
    y = coords[:, 0]
    x = coords[:, 1]
    x_norm = 2 * (x / (w - 1)) - 1
    y_norm = 2 * (y / (h - 1)) - 1
    grid = torch.stack([x_norm, y_norm], dim=1).view(1, 1, len(coords), 2)
    sampled = F.grid_sample(pred, grid, align_corners=True)
    return sampled.view(2, len(coords)).permute(1, 0)


@dataclass
class TrainConfig:
    batch_size: int = 2048
    lr: float = 1e-3
    num_steps: int = 10000
    sigma: float = 10.0
    log_interval: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    base_channels: int = 64
    amp: bool = True  # 混合精度加速
    seed: int = 42


def train_dsm(
    density_path: str | None = None,
    mask_path: str | None = None,
    model_out: str | None = None,
    cfg: TrainConfig | None = None,
):
    cfg = cfg or TrainConfig()
    density_np = np.load(density_path or config.TARGET_DENSITY_PATH)
    mask_np = np.load(mask_path or config.WALKABLE_MASK_PATH)

    total_samples = cfg.batch_size * cfg.num_steps
    dataset = ScoreDataset(
        density_np,
        mask_np,
        sigma=cfg.sigma,
        n_samples=min(total_samples, 2_000_000),  # 防止过大占内存
        seed=cfg.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    device = torch.device(cfg.device)
    static_map = torch.from_numpy(np.stack([mask_np, density_np], axis=0)).unsqueeze(0).float().to(device)
    model = UNetSmall(in_channels=2, base_channels=cfg.base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    def infinite_loader(dl):
        while True:
            for batch in dl:
                yield batch

    iterator = infinite_loader(loader)
    pbar = tqdm(range(cfg.num_steps), ncols=80, desc="train") if tqdm is not None else range(cfg.num_steps)

    last_loss = None
    for global_step in pbar:
        opt.zero_grad(set_to_none=True)
        target_vec, coords = next(iterator)
        target_vec = target_vec.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            pred_field = model(static_map)  # (1,2,H,W)
            pred_vec = _sample_pred_at_coords(pred_field, coords)
            loss = ((pred_vec - target_vec) ** 2).sum(dim=1).mean()

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        last_loss = float(loss.item())
        if tqdm is not None:
            pbar.set_postfix({"loss": f"{last_loss:.4f}"})
        elif global_step % cfg.log_interval == 0:
            print(f"[step {global_step}] loss={last_loss:.6f}")

    out_path = model_out or config.INNOVATION_MODEL_PATH
    torch.save(model.state_dict(), out_path)
    print(f"模型已保存: {out_path}")
    meta = asdict(cfg)
    meta.update(
        {
            "density_path": str(density_path or config.TARGET_DENSITY_PATH),
            "mask_path": str(mask_path or config.WALKABLE_MASK_PATH),
            "steps_finished": cfg.num_steps,
            "final_loss": last_loss,
        }
    )
    meta_path = str(out_path) + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"配置元数据已保存: {meta_path}")
    return model


def _parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train score UNet (DSM).")
    parser.add_argument("--density", type=str, default=None, help="path to target_density.npy")
    parser.add_argument("--mask", type=str, default=None, help="path to walkable_mask.npy")
    parser.add_argument("--out", type=str, default=None, help="output model path")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--no_amp", action="store_true", help="disable AMP")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = TrainConfig(
        batch_size=args.batch,
        lr=args.lr,
        num_steps=args.steps,
        sigma=args.sigma,
        device=args.device,
        num_workers=args.workers,
        base_channels=args.base_channels,
        amp=not args.no_amp,
        seed=args.seed,
    )
    return args, cfg


if __name__ == "__main__":
    args, cfg = _parse_cli_args()
    # One-Map 模式下，如 batch 过小则提升
    if cfg.batch_size < 1024:
        print(f"提示: One-Map 模式将 batch 从 {cfg.batch_size} 提升至 2048")
        cfg.batch_size = 2048
    train_dsm(
        density_path=args.density,
        mask_path=args.mask,
        model_out=args.out,
        cfg=cfg,
    )
