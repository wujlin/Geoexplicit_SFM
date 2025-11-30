"""
轻量 UNet，用于预测 score 场（2 通道）。
输入: 3 通道 [mask, density, query_map]
输出: 2 通道 [score_y, score_x]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32):
        super().__init__()
        self.inc = double_conv(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(base_channels * 2, base_channels * 4))
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv1 = double_conv(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv2 = double_conv(base_channels * 2, base_channels)
        self.outc = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x):
        h0, w0 = x.shape[-2], x.shape[-1]
        pad_h = (4 - h0 % 4) % 4
        pad_w = (4 - w0 % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        if x.shape[-2:] != x2.shape[-2:]:
            x2 = x2[..., : x.shape[-2], : x.shape[-1]]
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        if x.shape[-2:] != x1.shape[-2:]:
            x1 = x1[..., : x.shape[-2], : x.shape[-1]]
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        x = self.outc(x)
        if pad_h or pad_w:
            x = x[..., :h0, :w0]
        return x
