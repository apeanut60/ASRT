"""Generative / Low-derived structural Reference Prior Encoder."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from asrt.evaluation.reference_information import high_pass


def sobel_magnitude(img01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel Sobel magnitude. img01 [B,C,H,W] → [B,C,H,W]."""
    device, dtype = img01.device, img01.dtype
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    b, c, _, _ = img01.shape
    x = img01.reshape(b * c, 1, img01.shape[-2], img01.shape[-1])
    x = F.pad(x, (1, 1, 1, 1), mode='replicate')
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    return mag.reshape(b, c, img01.shape[-2], img01.shape[-1])


def build_structural_detail_input(
    low_m11: torch.Tensor,
    ref_m11: torch.Tensor,
    sigma: float = 7.0,
) -> torch.Tensor:
    """Z_R = cat[HP(R), Sobel(R), Sobel(X)] → [B,9,H,W]. Inputs [-1,1]."""
    x01 = ((low_m11 + 1.0) * 0.5).clamp(0, 1)
    r01 = ((ref_m11 + 1.0) * 0.5).clamp(0, 1)
    if tuple(r01.shape[-2:]) != tuple(x01.shape[-2:]):
        r01 = F.interpolate(r01, size=x01.shape[-2:], mode='bilinear', align_corners=False)
    h_r = high_pass(r01, sigma=float(sigma))
    g_r = sobel_magnitude(r01)
    g_x = sobel_magnitude(x01)
    return torch.cat([h_r, g_r, g_x], dim=1)


class ReferencePriorEncoder(nn.Module):
    """Encode 9-ch structural detail into H and H/2 C-channel priors."""

    def __init__(self, n_feats: int = 64, mid: int = 32, sigma: float = 7.0):
        super().__init__()
        self.n_feats = int(n_feats)
        self.sigma = float(sigma)
        self.stem = nn.Sequential(
            nn.Conv2d(9, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, n_feats, 3, padding=1),
            nn.GELU(),
        )
        self.to_h1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.down = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def encode_zr(self, zr: torch.Tensor) -> dict:
        f = self.stem(zr)
        h1 = self.to_h1(f)
        h2 = self.down(f)
        return dict(h1=h1, h2=h2)

    def forward(
        self,
        low_m11: torch.Tensor,
        ref_m11: Optional[torch.Tensor],
    ) -> Optional[dict]:
        if ref_m11 is None:
            return None
        zr = build_structural_detail_input(low_m11, ref_m11, sigma=self.sigma)
        return self.encode_zr(zr)
