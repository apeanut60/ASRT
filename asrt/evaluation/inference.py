"""Full-image and tiled inference. Global illum (if enabled) runs once on full image."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def infer_full(model, low, ref=None):
    model.eval()
    return model(low, ref=ref, return_aux=False)


@torch.no_grad()
def infer_tiled(model, low, ref=None, tile=256, overlap=32):
    """Tile restore; stitch with cosine feather. Ref tiles match Low tiles."""
    model.eval()
    # Temporarily disable global illum during tiles; apply once at end if needed.
    use_g = bool(getattr(model.backbone, 'use_global_illum', False))
    if use_g:
        model.backbone.use_global_illum = False

    _, _, H, W = low.shape
    if H <= tile and W <= tile:
        out = model(low, ref=ref)
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
            low_t = low[:, :, y0:y0 + tile, x0:x0 + tile]
            ref_t = None if ref is None else ref[:, :, y0:y0 + tile, x0:x0 + tile]
            pred = model(low_t, ref=ref_t)
            acc[:, :, y0:y0 + tile, x0:x0 + tile] += pred * feather
            wt[:, :, y0:y0 + tile, x0:x0 + tile] += feather
            if x0 + tile >= W:
                break
        if y0 + tile >= H:
            break
    out = acc / wt.clamp(min=1e-8)
    if use_g:
        model.backbone.use_global_illum = True
        out = model.backbone.global_illum_head.apply(out, low)
    return out
