"""
轻量 UNet，用于预测 score 场（2 通道）。
输入: 3 通道 [mask, density, query_map]
输出: 2 通道 [score_y, score_x]
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        x = self.outc(x)
        return x
