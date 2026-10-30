"""Lightweight trainable Reference Encoder for ASRT-v4."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h, inplace=True)


class RefEncoderV4(nn.Module):
    """RGB → F_ref_1 (H), F_ref_2 (H/2), F_ref_3 (H/4)."""

    def __init__(self, base_ch: int = 32, out_ch: int = 64):
        super().__init__()
        self.stem = _conv_bn_relu(3, base_ch, stride=1)
        self.rb1 = _ResBlock(base_ch)
        self.down1 = _conv_bn_relu(base_ch, base_ch * 2, stride=2)
        self.rb2 = _ResBlock(base_ch * 2)
        self.down2 = _conv_bn_relu(base_ch * 2, out_ch, stride=2)
        self.rb3 = _ResBlock(out_ch)
        self.proj1 = nn.Conv2d(base_ch, out_ch, 1)
        self.proj2 = nn.Conv2d(base_ch * 2, out_ch, 1)
        self.proj3 = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, ref_m11: torch.Tensor):
        """ref in [-1,1] → (F1,F2,F3) each with `out_ch` channels."""
        x = (ref_m11 + 1.0) * 0.5
        f1 = self.rb1(self.stem(x))
        f2 = self.rb2(self.down1(f1))
        f3 = self.rb3(self.down2(f2))
        return self.proj1(f1), self.proj2(f2), self.proj3(f3)
