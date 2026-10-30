"""Full-image and tiled inference. Global illum (if enabled) runs once on full image."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from asrt.models.query_illum_align import estimate_query_align_maps


def _full_image_query_maps(model, low, ref):
    """Estimate gamma/LUT once on full Low/Ref for tiled reuse."""
    cfg = getattr(model, 'query_align_cfg', {}) or {}
    if ref is None or not cfg.get('enabled', False):
        return None, None
    method = str(cfg.get('method', 'gamma_hist')).lower()
    if method in ('none', 'off', 'raw'):
        return None, None
    alpha = float(cfg.get('gamma_alpha', 0.75))
    gamma, lut, _ = estimate_query_align_maps(
        low, ref,
        alpha=alpha,
        gamma_min=float(cfg.get('gamma_min', 0.3)),
        gamma_max=float(cfg.get('gamma_max', 1.0)),
        bins=int(cfg.get('hist_bins', 256)),
    )
    return gamma, lut


@torch.no_grad()
def infer_full(model, low, ref=None):
    model.eval()
    # Full-image path: model estimates align internally if overrides are None.
    return model(low, ref=ref, return_aux=False)


def _v4_match_halo(model) -> int:
    """Extra context for local soft matching + shallow encoder RF (pixels)."""
    if not bool(getattr(model, 'is_v4', False)):
        return 0
    radius = 4
    matcher = getattr(model, 'matcher', None)
    if matcher is not None:
        radius = int(getattr(matcher, 'radius', radius))
    # H/4 search radius → full-res cells; + encoder RF margin
    return int(radius * 4 + 16)


@torch.no_grad()
def infer_tiled(model, low, ref=None, tile=256, overlap=32):
    """Tile restore; stitch with cosine feather. Ref tiles match Low tiles.

    Query alignment gamma/LUT estimated once on the full image and reused
    for every tile (stats_scope=full_image).

    For ASRTv4, each tile is expanded by a matching halo so local soft
    correspondence sees valid neighbors; only the core tile is blended.
    """
    model.eval()
    use_g = bool(getattr(model.backbone, 'use_global_illum', False))
    if use_g:
        model.backbone.use_global_illum = False

    qg, ql = _full_image_query_maps(model, low, ref)
    halo = _v4_match_halo(model)

    _, _, H, W = low.shape
    if H <= tile and W <= tile:
        out = model(low, ref=ref, query_gamma=qg, query_lut=ql)
        if use_g:
            model.backbone.use_global_illum = True
            out = model.backbone.global_illum_head.apply(out, low)
        return out

    stride = max(1, tile - overlap)
    acc = torch.zeros_like(low)
    wt = torch.zeros(1, 1, H, W, device=low.device)
    my = torch.ones(tile, device=low.device)
    mx = torch.ones(tile, device=low.device)
    if overlap > 0:
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, math.pi, overlap, device=low.device))
        my[:overlap] = ramp
        my[-overlap:] = ramp.flip(0)
        mx[:overlap] = ramp
        mx[-overlap:] = ramp.flip(0)
    feather = (my[:, None] * mx[None, :]).view(1, 1, tile, tile)

    for y0 in range(0, max(1, H - overlap), stride):
        for x0 in range(0, max(1, W - overlap), stride):
            y0 = min(y0, H - tile)
            x0 = min(x0, W - tile)
            y1, x1 = y0 + tile, x0 + tile
            ys = max(0, y0 - halo)
            xs = max(0, x0 - halo)
            ye = min(H, y1 + halo)
            xe = min(W, x1 + halo)
            low_t = low[:, :, ys:ye, xs:xe]
            ref_t = None if ref is None else ref[:, :, ys:ye, xs:xe]
            pred = model(low_t, ref=ref_t, query_gamma=qg, query_lut=ql)
            # Crop halo back to core tile
            cy0, cx0 = y0 - ys, x0 - xs
            pred_core = pred[:, :, cy0:cy0 + tile, cx0:cx0 + tile]
            acc[:, :, y0:y1, x0:x1] += pred_core * feather
            wt[:, :, y0:y1, x0:x1] += feather
            if x0 + tile >= W:
                break
        if y0 + tile >= H:
            break
    out = acc / wt.clamp(min=1e-8)
    if use_g:
        model.backbone.use_global_illum = True
        out = model.backbone.global_illum_head.apply(out, low)
    return out
