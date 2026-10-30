"""Dual reliability estimator for ASRT-v4."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualReliabilityEstimator(nn.Module):
    """Shared encoder + independent C_texture / C_illum heads at H/4."""

    def __init__(self, in_ch: int, mid: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.texture_head = nn.Conv2d(mid, 1, 1, bias=True)
        self.illum_head = nn.Conv2d(mid, 1, 1, bias=True)
        # mild init: prefer open gates early so adapters can learn
        nn.init.constant_(self.texture_head.bias, 0.0)
        nn.init.constant_(self.illum_head.bias, 0.0)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.encoder(feats)
        c_t = torch.sigmoid(self.texture_head(shared))
        c_l = torch.sigmoid(self.illum_head(shared))
        return c_t, c_l


def build_reliability_features(
    low_feat_h4: torch.Tensor,
    t_h4: torch.Tensor,
    match_diag: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Stack matching diagnostics + local structure + aligned-ref diff."""
    # local gradient magnitude of low feature (channel-mean)
    gx = low_feat_h4[:, :, :, 1:] - low_feat_h4[:, :, :, :-1]
    gy = low_feat_h4[:, :, 1:, :] - low_feat_h4[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    grad = (gx.abs().mean(dim=1, keepdim=True) + gy.abs().mean(dim=1, keepdim=True)) * 0.5
    feat_diff = (low_feat_h4 - t_h4).abs().mean(dim=1, keepdim=True)
    parts = [
        match_diag['entropy'],
        match_diag['disp_var'],
        match_diag['best_sim'],
        match_diag['exp_sim'],
        match_diag['continuity'],
        match_diag['max_weight'],
        match_diag['disp'][:, 0:1].abs(),
        match_diag['disp'][:, 1:2].abs(),
        grad,
        feat_diff,
    ]
    return torch.cat(parts, dim=1)


def upsample_gate(gate_h4: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(gate_h4, size=size, mode='bilinear', align_corners=True)
