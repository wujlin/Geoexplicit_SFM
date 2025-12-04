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
    """
    ResBlock with AdaLN (Adaptive Layer Normalization) conditioning.
    
    AdaLN 应用于 LayerNorm 之后: y = gamma * LN(x) + beta
    与 DiT (Diffusion Transformer) 使用相同的策略。
    
    关键点:
    1. condition 调制的是归一化后的特征，而非原始特征
    2. gamma bias 初始化为 1，weight 使用正常初始化
    3. 这确保初始时 gamma≈1（特征不被杀死），同时 condition 有合理的影响
    """
    def __init__(self, in_ch, out_ch, cond_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        # AdaLN: 分别输出 gamma 和 beta
        self.cond_gamma = nn.Linear(cond_ch, out_ch)
        self.cond_beta = nn.Linear(cond_ch, out_ch)
        
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        # 初始化策略:
        # gamma: bias=1, weight 正常初始化 → 初始 gamma ≈ 1 + small_variation
        # beta: bias=0, weight 正常初始化 → 初始 beta ≈ small_variation
        # 使用默认 Xavier/Kaiming 初始化即可，只需调整 bias
        nn.init.ones_(self.cond_gamma.bias)
        nn.init.zeros_(self.cond_beta.bias)

    def forward(self, x, cond):
        # x: (B, C, T), cond: (B, cond_ch)
        h = self.conv1(x)
        h = self.norm1(h)
        
        # AdaLN modulation
        gamma = self.cond_gamma(cond).unsqueeze(-1)  # (B, out_ch, 1)
        beta = self.cond_beta(cond).unsqueeze(-1)    # (B, out_ch, 1)
        h = gamma * h + beta
        
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
    """
    1D UNet for Diffusion Policy
    
    改进点:
    1. 增大默认通道数 64 -> 128
    2. 条件注入使用 AdaLN (Adaptive Layer Normalization)
       AdaLN: h = gamma * h + beta，gamma bias=1 确保初始时特征不被杀死
    3. time 和 obs 嵌入直接相加（标准 DDPM 做法）
    4. obs 嵌入使用可学习缩放因子（初始 2.0），确保 condition 有足够影响
    """
    def __init__(self, obs_dim: int = 12, act_dim: int = 2, base_channels: int = 128, 
                 cond_dim: int = 64, time_dim: int = 64):
        super().__init__()
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, cond_dim * 4),
            nn.GELU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )
        # 观测嵌入
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, cond_dim * 4),
            nn.GELU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )
        
        # obs 缩放因子：初始化为 2.0，确保 condition 有足够影响
        self.obs_scale = nn.Parameter(torch.tensor(2.0))

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
        global_cond: (B, obs_dim) 拼接的观测
        """
        # 时间嵌入
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        t_cond = self.time_embed(t_emb)  # (B, cond_dim)
        
        # 观测嵌入（带缩放）
        g_cond = self.obs_proj(global_cond) * self.obs_scale  # (B, cond_dim)
        
        # 条件融合: 直接相加
        cond = t_cond + g_cond  # (B, cond_dim)

        h = self.in_proj(x)
        h, s1 = self.down1(h, cond)
        h, s2 = self.down2(h, cond)
        h = self.mid(h, cond)
        h = self.up2(h, s2, cond)
        h = self.up1(h, s1, cond)
        out = self.out_proj(h)
        return out
