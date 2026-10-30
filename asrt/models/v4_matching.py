"""Geometry-constrained local soft matching for ASRT-v4."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSoftMatcher(nn.Module):
    """Local-window soft correspondence at H/4. No global HW×HW matrix."""

    def __init__(
        self,
        dim: int = 64,
        radius: int = 4,
        temperature: float = 0.1,
        lambda_pos: float = 0.5,
    ):
        super().__init__()
        self.dim = int(dim)
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.lambda_pos = float(lambda_pos)
        self.query_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
        )
        self.key_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
        )

    @property
    def window(self) -> int:
        return 2 * self.radius + 1

    def forward(
        self,
        low_feat_h4: torch.Tensor,
        illum_h4: torch.Tensor,
        ref_feat_h4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            T_h4: aggregated ref value [B,C,H,W]
            weight: soft weights [B, K, H, W] where K=window^2
            diag: matching diagnostics
        """
        b, _, h, w = low_feat_h4.shape
        ksz = self.window
        r = self.radius

        q_raw = self.query_proj(torch.cat([low_feat_h4, illum_h4], dim=1))
        k_raw = self.key_proj(ref_feat_h4)
        q_norm_pre = q_raw.norm(dim=1, keepdim=True)
        k_norm_pre = k_raw.norm(dim=1, keepdim=True)
        q = F.normalize(q_raw, dim=1, eps=1e-6)
        k = F.normalize(k_raw, dim=1, eps=1e-6)

        # Local key windows: [B, C*K, H, W] → [B, C, K, H, W]
        k_unf = F.unfold(k, kernel_size=ksz, padding=r)  # [B, C*K, HW]
        k_unf = k_unf.view(b, self.dim, ksz * ksz, h * w)

        q_flat = q.view(b, self.dim, 1, h * w)
        # cosine via channel sum
        sim = (q_flat * k_unf).sum(dim=1)  # [B, K, HW]

        # Relative offsets in feature cells
        yy, xx = torch.meshgrid(
            torch.arange(-r, r + 1, device=q.device, dtype=q.dtype),
            torch.arange(-r, r + 1, device=q.device, dtype=q.dtype),
            indexing='ij',
        )
        # normalized distance by radius
        dist2 = (yy * yy + xx * xx).reshape(1, ksz * ksz, 1) / float(max(r, 1) ** 2)
        score = sim / max(self.temperature, 1e-6) - self.lambda_pos * dist2

        # Mask out-of-bound candidates (unfold pads zeros; mark via ones mask)
        ones = torch.ones(b, 1, h, w, device=q.device, dtype=q.dtype)
        valid = F.unfold(ones, kernel_size=ksz, padding=r)  # [B, K, HW]
        score = score.masked_fill(valid < 0.5, -1e4)

        weight = torch.softmax(score, dim=1)  # [B, K, HW]
        weight = weight * valid
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # Aggregate ref values (use raw ref_feat as value, not key-proj)
        v_unf = F.unfold(ref_feat_h4, kernel_size=ksz, padding=r).view(
            b, ref_feat_h4.size(1), ksz * ksz, h * w)
        t = (v_unf * weight.unsqueeze(1)).sum(dim=2).view(b, ref_feat_h4.size(1), h, w)

        # Expected displacement (dx, dy) in feature cells
        dx = (weight * xx.reshape(1, -1, 1)).sum(dim=1).view(b, 1, h, w)
        dy = (weight * yy.reshape(1, -1, 1)).sum(dim=1).view(b, 1, h, w)
        disp = torch.cat([dx, dy], dim=1)

        # --- Diagnostics (masked / non-wrapping) ---
        n_valid = valid.sum(dim=1).clamp_min(1.0)  # [B, HW]
        ent_raw = -(weight.clamp_min(1e-8) * weight.clamp_min(1e-8).log()).sum(dim=1)
        # Normalize by log(#valid candidates); single-candidate → entropy 0
        log_n = n_valid.log()
        entropy = torch.where(
            n_valid <= 1.0 + 1e-6,
            torch.zeros_like(ent_raw),
            ent_raw / log_n.clamp_min(1e-8),
        ).view(b, 1, h, w)

        max_w, _ = weight.max(dim=1)
        max_w = max_w.view(b, 1, h, w)
        exp_sim = (weight * sim).sum(dim=1).view(b, 1, h, w)
        # Best similarity over *valid* candidates only
        sim_masked = sim.masked_fill(valid < 0.5, float('-inf'))
        best_sim = sim_masked.max(dim=1).values
        best_sim = torch.where(torch.isfinite(best_sim), best_sim, torch.zeros_like(best_sim))
        best_sim = best_sim.view(b, 1, h, w)

        ex2 = (weight * (xx.reshape(1, -1, 1) ** 2)).sum(dim=1).view(b, 1, h, w)
        ey2 = (weight * (yy.reshape(1, -1, 1) ** 2)).sum(dim=1).view(b, 1, h, w)
        var = (ex2 - dx * dx).clamp_min(0) + (ey2 - dy * dy).clamp_min(0)

        # Spatial continuity without toroidal wrap (no torch.roll)
        cont = torch.zeros(b, 1, h, w, device=disp.device, dtype=disp.dtype)
        cnt = torch.zeros_like(cont)
        if h > 1:
            d_v = (disp[:, :, 1:, :] - disp[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
            cont[:, :, :-1, :] = cont[:, :, :-1, :] + d_v
            cont[:, :, 1:, :] = cont[:, :, 1:, :] + d_v
            cnt[:, :, :-1, :] = cnt[:, :, :-1, :] + 1
            cnt[:, :, 1:, :] = cnt[:, :, 1:, :] + 1
        if w > 1:
            d_h = (disp[:, :, :, 1:] - disp[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
            cont[:, :, :, :-1] = cont[:, :, :, :-1] + d_h
            cont[:, :, :, 1:] = cont[:, :, :, 1:] + d_h
            cnt[:, :, :, :-1] = cnt[:, :, :, :-1] + 1
            cnt[:, :, :, 1:] = cnt[:, :, :, 1:] + 1
        cont = cont / cnt.clamp_min(1.0)

        weight_map = weight.view(b, ksz * ksz, h, w)
        diag = dict(
            weight=weight_map,
            disp=disp,
            entropy=entropy,
            max_weight=max_w,
            exp_sim=exp_sim,
            best_sim=best_sim,
            disp_var=var,
            continuity=cont,
            q_norm=q.norm(dim=1, keepdim=True),          # post-L2 (≈1)
            k_norm=k.norm(dim=1, keepdim=True),
            q_norm_pre=q_norm_pre,                        # pre-normalize
            k_norm_pre=k_norm_pre,
            n_valid=n_valid.view(b, 1, h, w),
        )
        return t, weight_map, diag


def displacement_to_grid(disp: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Convert feature-cell displacement [B,2,H,W] (dx,dy) to grid_sample grid at same HxW.

    grid_sample expects [-1,1] absolute coords. We start from identity grid then add
    normalized displacement (feature cells → [-1,1] span).
    """
    b, _, h, w = disp.shape
    device, dtype = disp.device, disp.dtype
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
    # one feature cell ≈ 2/max(H-1,1) in normalized coords
    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)
    # disp channels: dx (x/width), dy (y/height); grid is (x,y)
    delta = torch.stack([disp[:, 0] * sx * scale, disp[:, 1] * sy * scale], dim=1)
    grid = (base + delta).permute(0, 2, 3, 1)  # [B,H,W,2]
    return grid


def warp_by_disp(feat: torch.Tensor, disp_h4: torch.Tensor, scale: float) -> torch.Tensor:
    """Warp `feat` using H/4 displacement scaled to feat resolution.

    scale: feature-cell multiplier from H/4 to target (2 for H/2, 4 for H).
    """
    # Upsample displacement to feat size (in H/4 cell units), then multiply by scale
    # for pixel-equivalent... Our disp is in H/4 feature cells. At H/2, one H/4 cell
    # = 2 H/2 cells, so displace by disp*2 in H/2 grid cells.
    disp = F.interpolate(disp_h4, size=feat.shape[-2:], mode='bilinear', align_corners=True)
    disp = disp * float(scale)
    grid = displacement_to_grid(disp, scale=1.0)
    return F.grid_sample(
        feat, grid, mode='bilinear', padding_mode='border', align_corners=True)


def soft_aggregate_scalar(map_h4: torch.Tensor, weight: torch.Tensor, radius: int) -> torch.Tensor:
    """Aggregate a 1-channel (or C-channel) map at H/4 with soft match weights."""
    b, c, h, w = map_h4.shape
    ksz = 2 * radius + 1
    unf = F.unfold(map_h4, kernel_size=ksz, padding=radius).view(b, c, ksz * ksz, h * w)
    wflat = weight.view(b, ksz * ksz, h * w)
    out = (unf * wflat.unsqueeze(1)).sum(dim=2).view(b, c, h, w)
    return out
