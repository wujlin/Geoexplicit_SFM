"""
Score Field 训练（Denoising Score Matching）：
- 从 target_density 按概率采样点，加入高斯噪声
- 构造 query heatmap，网络输出 score 场
- 在噪声点位置处采样预测向量，与理论 score 目标做 MSE
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


def _sample_indices_from_density(density: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    flat = density.reshape(-1)
    flat = flat + 1e-9
    probs = flat / flat.sum()
    idx = rng.choice(len(flat), p=probs)
    h, w = density.shape
    y = idx // w
    x = idx % w
    return int(y), int(x)


class ScoreDataset(Dataset):
    def __init__(
        self,
        density: np.ndarray,
        mask: np.ndarray,
        sigma: float = 10.0,
        n_samples: int = 50000,
        seed: int = 42,
    ):
        self.density = density.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.sigma = sigma
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.h, self.w = density.shape
        self.static_input = np.stack([self.mask, self.density], axis=0)

    def __len__(self):
        return self.n_samples

    def reset_rng(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        y0, x0 = _sample_indices_from_density(self.density, self.rng)
        epsilon = self.rng.normal(loc=0.0, scale=1.0, size=2).astype(np.float32)
        y_noisy = float(np.clip(y0 + epsilon[0] * self.sigma, 0, self.h - 1))
        x_noisy = float(np.clip(x0 + epsilon[1] * self.sigma, 0, self.w - 1))

        # 预测噪声本身（带负号表示恢复方向），避免数值过小坍缩
        target_vec = -1.0 * epsilon
        coord = np.array([y_noisy, x_noisy], dtype=np.float32)
        return torch.from_numpy(self.static_input), torch.from_numpy(target_vec), torch.from_numpy(coord)


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
    num_steps: int = 5000
    sigma: float = 10.0
    log_interval: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    base_channels: int = 32
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

    dataset = ScoreDataset(
        density_np,
        mask_np,
        sigma=cfg.sigma,
        n_samples=cfg.num_steps * cfg.batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=(lambda worker_id: dataset.reset_rng(cfg.seed + worker_id)) if cfg.num_workers > 0 else None,
    )

    device = torch.device(cfg.device)
    model = UNetSmall(in_channels=2, base_channels=cfg.base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=min(cfg.num_steps, len(loader)), ncols=80, desc="train")

    global_step = 0
    last_loss = None
    for batch in iterator:
        opt.zero_grad(set_to_none=True)
        inp, target_vec, coords = batch
        inp = inp.to(device, non_blocking=True)
        target_vec = target_vec.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            pred_field = model(inp)
            pred_vec = _sample_pred_at_coords(pred_field, coords)
            loss = ((pred_vec - target_vec) ** 2).sum(dim=1).mean()

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        last_loss = float(loss.item())
        if global_step % cfg.log_interval == 0:
            print(f"[step {global_step}] loss={last_loss:.6f}")
        if tqdm is not None:
            iterator.set_postfix({"loss": f"{last_loss:.4f}"})
        global_step += 1
        if global_step >= cfg.num_steps:
            break

    out_path = model_out or config.INNOVATION_MODEL_PATH
    torch.save(model.state_dict(), out_path)
    print(f"模型已保存: {out_path}")
    meta = asdict(cfg)
    meta.update(
        {
            "density_path": str(density_path or config.TARGET_DENSITY_PATH),
            "mask_path": str(mask_path or config.WALKABLE_MASK_PATH),
            "steps_finished": global_step,
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
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=32)
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
    train_dsm(
        density_path=args.density,
        mask_path=args.mask,
        model_out=args.out,
        cfg=cfg,
    )
