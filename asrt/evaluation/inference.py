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


def _feather_2d(th, tw, y0, x0, H, W, overlap, device):
    """Cosine feather only on *interior* tile overlaps; image borders stay 1."""
    my = torch.ones(th, device=device)
    mx = torch.ones(tw, device=device)
    if overlap > 0:
        n = min(overlap, th, tw)
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, math.pi, n, device=device))
        if y0 > 0:
            my[:n] = ramp
        if y0 + th < H:
            my[-n:] = ramp.flip(0)
        if x0 > 0:
            mx[:n] = ramp
        if x0 + tw < W:
            mx[-n:] = ramp.flip(0)
    return (my[:, None] * mx[None, :]).view(1, 1, th, tw)


@torch.no_grad()
def infer_tiled(model, low, ref=None, tile=256, overlap=32, return_weight=False):
    """Tile restore; stitch with cosine feather. Ref tiles match Low tiles.

    - Image outer borders keep weight 1 (no zero-ramp at canvas edge).
    - Only inter-tile overlaps are feathered.
    - Supports H>tile & W<tile (and vice versa) via dynamic tile sizes.
    - ASRTv4: each crop expands by matching halo; only core is blended.
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
        if return_weight:
            return out, torch.ones(1, 1, H, W, device=low.device)
        return out

    tile_h = min(int(tile), H)
    tile_w = min(int(tile), W)
    overlap_y = min(int(overlap), max(0, tile_h - 1))
    overlap_x = min(int(overlap), max(0, tile_w - 1))
    stride_y = max(1, tile_h - overlap_y)
    stride_x = max(1, tile_w - overlap_x)

    acc = torch.zeros_like(low)
    wt = torch.zeros(1, 1, H, W, device=low.device)

    y_starts = list(range(0, max(1, H - tile_h + 1), stride_y))
    if y_starts[-1] != H - tile_h:
        y_starts.append(H - tile_h)
    x_starts = list(range(0, max(1, W - tile_w + 1), stride_x))
    if x_starts[-1] != W - tile_w:
        x_starts.append(W - tile_w)
    # Deduplicate while preserving order
    y_starts = list(dict.fromkeys(max(0, y) for y in y_starts))
    x_starts = list(dict.fromkeys(max(0, x) for x in x_starts))

    for y0 in y_starts:
        for x0 in x_starts:
            assert y0 >= 0 and x0 >= 0
            y1, x1 = y0 + tile_h, x0 + tile_w
            assert y1 <= H and x1 <= W
            ys = max(0, y0 - halo)
            xs = max(0, x0 - halo)
            ye = min(H, y1 + halo)
            xe = min(W, x1 + halo)
            low_t = low[:, :, ys:ye, xs:xe]
            ref_t = None if ref is None else ref[:, :, ys:ye, xs:xe]
            pred = model(low_t, ref=ref_t, query_gamma=qg, query_lut=ql)
            cy0, cx0 = y0 - ys, x0 - xs
            pred_core = pred[:, :, cy0:cy0 + tile_h, cx0:cx0 + tile_w]
            if pred_core.shape[-2:] != (tile_h, tile_w):
                raise RuntimeError(
                    'tile core shape %s != (%d,%d)' % (pred_core.shape[-2:], tile_h, tile_w))
            feather = _feather_2d(
                tile_h, tile_w, y0, x0, H, W,
                max(overlap_y, overlap_x), low.device)
            acc[:, :, y0:y1, x0:x1] += pred_core * feather
            wt[:, :, y0:y1, x0:x1] += feather

    if bool((wt <= 0).any()):
        raise RuntimeError(
            'tiled inference left uncovered pixels: zero_weight=%d / %d'
            % (int((wt <= 0).sum()), int(wt.numel())))
    out = acc / wt
    if use_g:
        model.backbone.use_global_illum = True
        out = model.backbone.global_illum_head.apply(out, low)
    if return_weight:
        return out, wt
    return out
