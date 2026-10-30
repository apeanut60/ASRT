"""Reference-conditioned ResBlock with zero-init spatial affine modulation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)


class CondResBlock(nn.Module):
    """ResBlock + optional Spatial Affine Modulation from L_guided.

    When L_guided/C_L is None, identical to ResBlock (adapter unused).
    Adapter last layer zero-init → initial forward matches plain ResBlock.
    """

    def __init__(self, channels: int, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        # gamma, beta from L_guided
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 1, bias=True),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x, l_guided=None, c_l=None):
        h = self.relu(self.conv1(x))
        if l_guided is not None and c_l is not None:
            if l_guided.shape[-2:] != h.shape[-2:]:
                l_guided = F.interpolate(
                    l_guided, size=h.shape[-2:], mode='bilinear', align_corners=False)
            if c_l.shape[-2:] != h.shape[-2:]:
                c_l = F.interpolate(
                    c_l, size=h.shape[-2:], mode='bilinear', align_corners=False)
            gb = self.adapter(l_guided)
            gamma, beta = gb.chunk(2, dim=1)
            h = h * (1.0 + c_l * gamma) + c_l * beta
        return x + self.res_scale * self.conv2(h)
