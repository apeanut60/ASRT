"""Texture / Illumination adapters for ASRT-v4 (zero-init residual)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_init_last(conv: nn.Conv2d) -> None:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class TextureAdapter(nn.Module):
    """delta_f = adapter(concat(F, T)); F' = F + C * delta_f."""

    def __init__(self, feat_ch: int, mid: int | None = None):
        super().__init__()
        mid = mid or max(feat_ch, 32)
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch * 2, mid, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, feat_ch, 3, padding=1, bias=True),
        )
        _zero_init_last(self.net[-1])

    def forward(self, f: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([f, t], dim=1))
        return f + c * delta


class IlluminationAdapter(nn.Module):
    """delta_L = adapter(concat(L, P_ref, D_ref)); L' = L + C * delta_L."""

    def __init__(self, illum_ch: int, mid: int | None = None):
        super().__init__()
        mid = mid or max(illum_ch, 32)
        # L (C) + P_ref (1) + D_ref (1)
        self.net = nn.Sequential(
            nn.Conv2d(illum_ch + 2, mid, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, illum_ch, 3, padding=1, bias=True),
        )
        _zero_init_last(self.net[-1])

    def forward(
        self,
        l: torch.Tensor,
        p_ref: torch.Tensor,
        d_ref: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        if p_ref.shape[-2:] != l.shape[-2:]:
            p_ref = F.interpolate(p_ref, size=l.shape[-2:], mode='bilinear', align_corners=True)
        if d_ref.shape[-2:] != l.shape[-2:]:
            d_ref = F.interpolate(d_ref, size=l.shape[-2:], mode='bilinear', align_corners=True)
        if c.shape[-2:] != l.shape[-2:]:
            c = F.interpolate(c, size=l.shape[-2:], mode='bilinear', align_corners=True)
        delta = self.net(torch.cat([l, p_ref, d_ref], dim=1))
        return l + c * delta


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    # x in [0,1], RGB
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class LowPassLuminance(nn.Module):
    """Depthwise Gaussian-ish low-pass on luminance."""

    def __init__(self, ksize: int = 9, sigma: float = 3.0):
        super().__init__()
        assert ksize % 2 == 1
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize // 2)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        ker = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        ker = ker / ker.sum()
        self.register_buffer('kernel', ker.view(1, 1, ksize, ksize), persistent=False)
        self.pad = ksize // 2

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        lum = rgb_to_luminance(rgb)
        return F.conv2d(lum, self.kernel, padding=self.pad)
