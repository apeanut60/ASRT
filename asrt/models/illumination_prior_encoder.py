"""ISALux-style Illumination Prior from Low only: P_I = 1 - max_c(X)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def illumination_prior_map(low_m11: torch.Tensor) -> torch.Tensor:
    """Low [-1,1] → P_I [B,1,H,W] in [0,1]."""
    x01 = (low_m11 + 1.0) * 0.5
    return 1.0 - x01.amax(dim=1, keepdim=True)


class IlluminationPriorEncoder(nn.Module):
    """Project 1-ch illumination map to multi-scale C-channel priors."""

    def __init__(self, n_feats: int = 64):
        super().__init__()
        self.n_feats = int(n_feats)
        self.proj_h1 = nn.Sequential(
            nn.Conv2d(1, self.n_feats, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.n_feats, self.n_feats, 3, padding=1),
        )
        self.proj_h2 = nn.Sequential(
            nn.Conv2d(1, self.n_feats, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.n_feats, self.n_feats, 3, padding=1),
        )
        self.proj_h4 = nn.Sequential(
            nn.Conv2d(1, self.n_feats, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.n_feats, self.n_feats, 3, padding=1),
        )

    def forward(self, low_m11: torch.Tensor) -> dict:
        p = illumination_prior_map(low_m11)
        _, _, h, w = p.shape
        p_h2 = F.interpolate(p, size=(max(1, h // 2), max(1, w // 2)),
                             mode='bilinear', align_corners=False)
        p_h4 = F.interpolate(p, size=(max(1, h // 4), max(1, w // 4)),
                             mode='bilinear', align_corners=False)
        return dict(
            h1=self.proj_h1(p),
            h2=self.proj_h2(p_h2),
            h4=self.proj_h4(p_h4),
            map=p,
        )
