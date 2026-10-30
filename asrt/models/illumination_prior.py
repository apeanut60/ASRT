"""Matched Reference Illumination Prior + shared Illumination Adapter."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_luminance01(rgb_m11: torch.Tensor) -> torch.Tensor:
    """RGB [-1,1] → luminance [0,1], shape [B,1,H,W]."""
    rgb = (rgb_m11 + 1.0) * 0.5
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def lowpass(x: torch.Tensor, k: int = 15) -> torch.Tensor:
    """Separable box low-pass (odd k)."""
    if k < 3:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    w = x.new_ones(1, 1, 1, k) / float(k)
    y = F.conv2d(x, w, padding=(0, pad))
    w2 = x.new_ones(1, 1, k, 1) / float(k)
    return F.conv2d(y, w2, padding=(pad, 0))


def warp_by_match_index(feat: torch.Tensor, match_index: torch.Tensor, h4: int, w4: int) -> torch.Tensor:
    """Gather feat at H/4 grid using flattened match indices from SearchTransfer.

    feat: [B,C,H_ref,W_ref] — should already be at H/4 (or resized to h4,w4)
    match_index: [B, h4*w4] long indices into flattened H/4 ref grid
    Returns: [B,C,h4,w4]
    """
    b, c, _, _ = feat.shape
    if feat.shape[-2:] != (h4, w4):
        feat = F.interpolate(feat, size=(h4, w4), mode='bilinear', align_corners=False)
    flat = feat.view(b, c, h4 * w4)
    idx = match_index.view(b, 1, h4 * w4).expand(-1, c, -1)
    gathered = torch.gather(flat, 2, idx)
    return gathered.view(b, c, h4, w4)


def build_matched_illumination(
    ref_m11: torch.Tensor,
    base_m11: torch.Tensor,
    match_index: torch.Tensor,
    h4: int,
    w4: int,
    lpf_k: int = 15,
    eps: float = 1e-3,
    d_clamp: float = 2.0,
):
    """Build P_ref_matched, P_base, D_ref (local) at H/4.

    P_* are low-frequency luminance proxies in [0,1].
    D_ref separates global mean so Adapter cannot freely shift whole-image brightness.
    """
    p_ref = lowpass(rgb_luminance01(ref_m11), k=lpf_k)
    p_base_full = lowpass(rgb_luminance01(base_m11), k=lpf_k)
    p_ref_m = warp_by_match_index(p_ref, match_index, h4, w4)
    # Soften hard-argmax blockiness
    p_ref_m = lowpass(p_ref_m, k=max(5, lpf_k // 2))
    p_base = F.interpolate(p_base_full, size=(h4, w4), mode='bilinear', align_corners=False)

    log_r = torch.log(p_ref_m.clamp_min(eps))
    log_b = torch.log(p_base.clamp_min(eps))
    d = (log_r - log_b).clamp(-d_clamp, d_clamp)
    d_mean = d.mean(dim=(2, 3), keepdim=True)
    d_local = d - d_mean  # relative local difference only
    return dict(
        p_ref_matched=p_ref_m,
        p_base=p_base,
        d_ref=d_local,
        d_ref_global=d_mean,
        p_ref_full_lpf=p_ref,
    )


class IlluminationAdapter(nn.Module):
    """DeltaL_ref with zero-init last layer: L_guided = L_low + C_L * DeltaL."""

    def __init__(self, n_feats: int):
        super().__init__()
        # L_low (C) + P_ref + P_base + D_ref + C_L = C+4
        self.conv1 = nn.Conv2d(n_feats + 4, n_feats, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(
        self,
        l_low: torch.Tensor,
        p_ref: torch.Tensor,
        p_base: torch.Tensor,
        d_ref: torch.Tensor,
        c_l: torch.Tensor,
    ):
        def _rz(t):
            return t if t.shape[-2:] == l_low.shape[-2:] else F.interpolate(
                t, size=l_low.shape[-2:], mode='bilinear', align_corners=False)

        p_ref, p_base, d_ref, c_l = map(_rz, (p_ref, p_base, d_ref, c_l))
        x = torch.cat([l_low, p_ref, p_base, d_ref, c_l], dim=1)
        delta = self.conv2(self.relu(self.conv1(x)))
        l_guided = l_low + c_l * delta
        return l_guided, delta
