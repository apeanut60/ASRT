"""Full-image and tiled inference. Global illum (if enabled) runs once on full image."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from asrt.models.prior_ops import prior_context_halo
from asrt.models.query_illum_align import estimate_query_align_maps


def is_irp_model(model) -> bool:
    if bool(getattr(model, 'is_irp', False)):
        return True
    fam = str(getattr(model, 'model_family', '')).lower()
    return fam in ('asrt_irp', 'irp', 'irp_v1')


def _full_image_query_maps(model, low, ref):
    """Estimate gamma/LUT once on full Low/Ref for tiled reuse (legacy/v3/v4)."""
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


def call_model(
    model,
    low,
    ref=None,
    query_gamma=None,
    query_lut=None,
    return_aux: bool = False,
):
    """Dispatch forward by explicit model capability (no silent TypeError retry)."""
    if is_irp_model(model):
        # IRP does not accept TTSR query alignment kwargs.
        return model(low, ref=ref, return_aux=return_aux)
    return model(
        low, ref=ref, return_aux=return_aux,
        query_gamma=query_gamma, query_lut=query_lut)


def _v4_match_halo(model) -> int:
    """Extra context for local soft matching + shallow encoder RF (pixels)."""
    if not bool(getattr(model, 'is_v4', False)):
        return 0
    radius = 4
    matcher = getattr(model, 'matcher', None)
    if matcher is not None:
        radius = int(getattr(matcher, 'radius', radius))
    return int(radius * 4 + 16)


def _irp_tile_halo(model) -> int:
    """Context for IRP tiled crops: Gaussian HP + Sobel + one attention window."""
    sigma = float(getattr(model, 'ref_sigma', 7.0))
    ws = int(getattr(model, 'window_size', 8))
    # CSFI / ResBlock RF margin (conservative, not claimed exact)
    csfi_margin = 16
    return int(prior_context_halo(sigma=sigma, sobel=1) + ws + csfi_margin)


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


def _aligned_starts(length: int, tile: int, stride: int, align: int) -> list:
    """Tile starts snapped to a global grid of size `align` (e.g. window_size)."""
    align = max(1, int(align))
    tile = min(int(tile), length)
    if length <= tile:
        return [0]
    stride = max(align, (int(stride) // align) * align)
    if stride < 1:
        stride = align
    starts = []
    y = 0
    while y + tile < length:
        starts.append(y)
        y += stride
    last = length - tile
    # snap last start down to align grid
    last_aligned = (last // align) * align
    if last_aligned not in starts:
        starts.append(last_aligned)
    # Ensure coverage to the end: if snapped last leaves a gap, append exact last
    # (may break perfect window alignment at the right/bottom border — reported).
    if starts[-1] != last and starts[-1] + tile < length:
        starts.append(last)
    return list(dict.fromkeys(max(0, s) for s in starts))


@torch.no_grad()
def infer_full(model, low, ref=None):
    model.eval()
    return call_model(model, low, ref=ref, return_aux=False)


@torch.no_grad()
def infer_tiled(
    model,
    low,
    ref=None,
    tile=256,
    overlap=32,
    return_weight=False,
    allow_unaligned_border: bool = True,
):
    """Tile restore; stitch with cosine feather. Ref tiles match Low tiles.

    IRP path:
      - No query_gamma/lut kwargs.
      - Halo sized for Gaussian HP + Sobel + window attention.
      - Tile starts snapped to window_size grid when possible.
      - Full/Tiled are NOT claimed bit-exact; callers must measure error.
    """
    model.eval()
    use_g = bool(getattr(getattr(model, 'backbone', None), 'use_global_illum', False))
    if use_g:
        model.backbone.use_global_illum = False

    irp = is_irp_model(model)
    qg = ql = None
    if not irp:
        qg, ql = _full_image_query_maps(model, low, ref)
        halo = _v4_match_halo(model)
        align = 1
    else:
        halo = _irp_tile_halo(model)
        align = int(getattr(model, 'window_size', 8))

    _, _, H, W = low.shape
    if H <= tile and W <= tile:
        out = call_model(model, low, ref=ref, query_gamma=qg, query_lut=ql)
        if use_g:
            model.backbone.use_global_illum = True
            out = model.backbone.global_illum_head.apply(out, low)
        if return_weight:
            return out, torch.ones(1, 1, H, W, device=low.device)
        return out

    tile_h = min(int(tile), H)
    tile_w = min(int(tile), W)
    # Snap tile size to align grid when possible (IRP window origin)
    if irp and align > 1:
        tile_h = max(align, (tile_h // align) * align)
        tile_w = max(align, (tile_w // align) * align)
        tile_h = min(tile_h, H)
        tile_w = min(tile_w, W)
    overlap_y = min(int(overlap), max(0, tile_h - 1))
    overlap_x = min(int(overlap), max(0, tile_w - 1))
    if irp and align > 1:
        # keep overlap a multiple of window size so strides stay aligned
        overlap_y = (overlap_y // align) * align
        overlap_x = (overlap_x // align) * align
    stride_y = max(1, tile_h - overlap_y)
    stride_x = max(1, tile_w - overlap_x)

    acc = torch.zeros_like(low)
    wt = torch.zeros(1, 1, H, W, device=low.device)

    if irp:
        y_starts = _aligned_starts(H, tile_h, stride_y, align)
        x_starts = _aligned_starts(W, tile_w, stride_x, align)
    else:
        y_starts = list(range(0, max(1, H - tile_h + 1), stride_y))
        if y_starts[-1] != H - tile_h:
            y_starts.append(H - tile_h)
        x_starts = list(range(0, max(1, W - tile_w + 1), stride_x))
        if x_starts[-1] != W - tile_w:
            x_starts.append(W - tile_w)
        y_starts = list(dict.fromkeys(max(0, y) for y in y_starts))
        x_starts = list(dict.fromkeys(max(0, x) for x in x_starts))

    for y0 in y_starts:
        for x0 in x_starts:
            assert y0 >= 0 and x0 >= 0
            # dynamic core size near borders
            th = min(tile_h, H - y0)
            tw = min(tile_w, W - x0)
            y1, x1 = y0 + th, x0 + tw
            assert y1 <= H and x1 <= W
            ys = max(0, y0 - halo)
            xs = max(0, x0 - halo)
            ye = min(H, y1 + halo)
            xe = min(W, x1 + halo)
            low_t = low[:, :, ys:ye, xs:xe]
            ref_t = None if ref is None else ref[:, :, ys:ye, xs:xe]
            pred = call_model(model, low_t, ref=ref_t, query_gamma=qg, query_lut=ql)
            cy0, cx0 = y0 - ys, x0 - xs
            pred_core = pred[:, :, cy0:cy0 + th, cx0:cx0 + tw]
            if pred_core.shape[-2:] != (th, tw):
                raise RuntimeError(
                    'tile core shape %s != (%d,%d)' % (pred_core.shape[-2:], th, tw))
            feather = _feather_2d(
                th, tw, y0, x0, H, W,
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


def full_vs_tiled_stats(
    model,
    low,
    ref=None,
    tile=256,
    overlap=32,
) -> dict:
    """Quantify Full vs Tiled pixel error (not claimed bit-exact for IRP)."""
    full = infer_full(model, low, ref)
    tiled, wt = infer_tiled(
        model, low, ref, tile=tile, overlap=overlap, return_weight=True)
    diff = (full - tiled).abs()
    return dict(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        rmse=float(torch.sqrt((diff ** 2).mean()).item()),
        uncovered=int((wt <= 0).sum().item()),
        is_irp=is_irp_model(model),
        tile=int(tile),
        overlap=int(overlap),
        halo=_irp_tile_halo(model) if is_irp_model(model) else _v4_match_halo(model),
    )
