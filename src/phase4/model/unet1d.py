"""
1D Conditional UNet，用于 Diffusion Policy：
- 输入: noisy action 序列 (B, T, C=2)
- 条件: timestep embedding + global condition (如目标方向/其他特征)
- 输出: 与输入同形状的去噪预测
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    标准时间步嵌入，返回 (B, dim)
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.cond_proj = nn.Linear(cond_ch, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        # x: (B, C, T), cond: (B, cond_ch)
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        return h + self.skip(x)


class Down1D(nn.Module):
    def __init__(self, ch, cond_ch):
        super().__init__()
        self.block = ResBlock1D(ch, ch, cond_ch)
        self.down = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, cond):
        x = self.block(x, cond)
        skip = x
        x = self.down(x)
        return x, skip


class Up1D(nn.Module):
    def __init__(self, ch, cond_ch):
        super().__init__()
        self.block = ResBlock1D(ch * 2, ch, cond_ch)
        self.up = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, cond)
        return x


class UNet1D(nn.Module):
    def __init__(self, obs_dim: int = 4, act_dim: int = 2, base_channels: int = 64, cond_dim: int = 16, time_dim: int = 32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.ReLU(),
            nn.Linear(time_dim * 4, cond_dim),
        )
        self.obs_proj = nn.Linear(obs_dim, cond_dim)

        self.in_proj = nn.Conv1d(act_dim, base_channels, kernel_size=3, padding=1)
        self.down1 = Down1D(base_channels, cond_dim)
        self.down2 = Down1D(base_channels, cond_dim)
        self.mid = ResBlock1D(base_channels, base_channels, cond_dim)
        self.up2 = Up1D(base_channels, cond_dim)
        self.up1 = Up1D(base_channels, cond_dim)
        self.out_proj = nn.Conv1d(base_channels, act_dim, kernel_size=3, padding=1)

        self.time_dim = time_dim
        self.cond_dim = cond_dim

    def forward(self, x, timesteps, global_cond):
        """
        x: (B, act_dim, T) noisy action
        timesteps: (B,) int64
        global_cond: (B, obs_dim) 例如拼接的观测（可选）
        """
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        t_cond = self.time_embed(t_emb)  # (B, cond_dim)
        g_cond = self.obs_proj(global_cond)  # (B, cond_dim)
        cond = t_cond + g_cond

        h = self.in_proj(x)
        h, s1 = self.down1(h, cond)
        h, s2 = self.down2(h, cond)
        h = self.mid(h, cond)
        h = self.up2(h, s2, cond)
        h = self.up1(h, s1, cond)
        out = self.out_proj(h)
        return out
