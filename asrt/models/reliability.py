"""Dual Reliability Estimator: C_T (texture) and C_L (illumination)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_grad_mag(x01: torch.Tensor) -> torch.Tensor:
    """x01: [B,3,H,W] or [B,1,H,W] in [0,1] → [B,1,H,W] gradient magnitude."""
    if x01.size(1) == 3:
        g = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
    else:
        g = x01
    kx = g.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = g.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def local_variance(x01: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Smoothed local variance of luminance. Avoids NaN when var≈0."""
    if x01.size(1) == 3:
        g = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
    else:
        g = x01
    pad = k // 2
    mean = F.avg_pool2d(g, k, stride=1, padding=pad)
    mean2 = F.avg_pool2d(g * g, k, stride=1, padding=pad)
    var = (mean2 - mean * mean).clamp_min(0.0)
    return var


def low_structure_maps(low_m11: torch.Tensor, out_hw) -> torch.Tensor:
    """Return [B,2,h,w] = (grad_mag, local_var) at target resolution."""
    low01 = (low_m11 + 1.0) * 0.5
    grad = _sobel_grad_mag(low01)
    var = local_variance(low01, k=7)
    # Soft-normalize per-image to reduce extreme dark noise dominance
    def _norm(t):
        flat = t.float().flatten(2)
        p95 = flat.quantile(0.95, dim=-1, keepdim=True).clamp_min(1e-4)
        return (t / p95.view(-1, 1, 1, 1).to(dtype=t.dtype)).clamp(0.0, 1.0)

    feat = torch.cat([_norm(grad), _norm(var)], dim=1)
    if tuple(feat.shape[-2:]) != tuple(out_hw):
        feat = F.interpolate(feat, size=out_hw, mode='bilinear', align_corners=False)
    return feat


class DualReliabilityEstimator(nn.Module):
    """Lightweight CNN → C_T, C_L in [0,1] at H/4.

    Init: C_T bias high (near-open texture gate); C_L mid.
    Illumination effect is gated by zero-init Adapter elsewhere.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        # match_score, margin, cycle, spatial + grad, var = 6
        in_ch = 6
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.head_t = nn.Conv2d(hidden, 1, 1, bias=True)
        self.head_l = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.zeros_(self.head_t.weight)
        nn.init.constant_(self.head_t.bias, 2.0)  # sigmoid(2)≈0.88
        nn.init.zeros_(self.head_l.weight)
        nn.init.zeros_(self.head_l.bias)  # sigmoid(0)=0.5

    def forward(self, match_diag: dict, low_m11: torch.Tensor):
        score = match_diag['match_score']
        hw = score.shape[-2:]
        struct = low_structure_maps(low_m11, hw)
        x = torch.cat([
            score,
            match_diag['match_margin'],
            match_diag['cycle_consistency'],
            match_diag['spatial_consistency'],
            struct,
        ], dim=1)
        h = self.encoder(x)
        c_t = torch.sigmoid(self.head_t(h))
        c_l = torch.sigmoid(self.head_l(h))
        return c_t, c_l


class FineCTRefine(nn.Module):
    """Light high-res correction of upsampled C_T using Low structure."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1 + 2, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c_t_up: torch.Tensor, low_m11: torch.Tensor):
        struct = low_structure_maps(low_m11, c_t_up.shape[-2:])
        delta = self.net(torch.cat([c_t_up, struct], dim=1))
        out = torch.sigmoid(
            torch.logit(c_t_up.clamp(1e-4, 1 - 1e-4)) + delta)
        # Preserve hard zeros/ones so C_T=0 ⇒ texture residual exactly 0
        out = torch.where(c_t_up <= 0, torch.zeros_like(out), out)
        out = torch.where(c_t_up >= 1, torch.ones_like(out), out)
        return out


def reliability_stats(c: torch.Tensor) -> dict:
    flat = c.detach().float().flatten()
    qs = torch.quantile(
        flat, torch.tensor([0.1, 0.5, 0.9], device=flat.device, dtype=flat.dtype))
    return dict(
        mean=float(flat.mean()),
        std=float(flat.std(unbiased=False)),
        q10=float(qs[0]),
        q50=float(qs[1]),
        q90=float(qs[2]),
        min=float(flat.min()),
        max=float(flat.max()),
    )
