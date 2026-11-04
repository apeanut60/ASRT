"""Halo-aware synchronized crop for IRP training.

Attention window partition uses the *outer* (halo-inclusive) crop origin as
(0,0). To keep H / H/2 / H/4 window phases aligned with a global lattice under
exact factor-2 spatial mapping, outer crop origins use:

    alignment = window_size * 4   # e.g. 32 when window_size=8

Halo is rounded up so:
    halo >= prior_context (Gaussian radius + Sobel)
    halo % alignment == 0

Then:
    outer_top % alignment == 0
    valid_top = outer_top + halo   (also aligned when halo % alignment == 0)

Note: CSFI uses stride-2 Conv with padding=1, which is not a pure G→G/2 map;
exact bit-match to full-image windows is not claimed. This alignment removes
the dominant phase offset from outer≠valid origins.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from asrt.models.prior_ops import prior_context_halo


def window_grid_alignment(window_size: int = 8) -> int:
    """Full-res crop alignment for H + H/2 + H/4 window lattices."""
    return max(1, int(window_size) * 4)


def compute_irp_halo(
    sigma: float = 7.0,
    window_size: int = 8,
    sobel: int = 1,
    extra: int = 0,
) -> int:
    """Halo ≥ prior context, rounded up to window_grid_alignment."""
    base = int(prior_context_halo(sigma=sigma, sobel=sobel) + int(extra))
    align = window_grid_alignment(window_size)
    if base <= 0:
        return align
    return int(math.ceil(base / float(align)) * align)


def sync_geometry_halo(
    tensors: Sequence[torch.Tensor],
    crop: int,
    rng: np.random.RandomState,
    halo: int,
    window_size: int = 8,
    align_valid_to_window: bool = True,
) -> Dict[str, object]:
    """Shared random crop with halo context + flips/rotations.

    When align_valid_to_window=True (default for IRP train):
      outer_top/left snapped to alignment = window_size*4
      valid_* = outer_* + halo
    """
    assert len(tensors) >= 1
    halo = max(0, int(halo))
    crop = int(crop)
    _, h0, w0 = tensors[0].shape
    need = crop + 2 * halo
    pad_h = max(0, need - h0)
    pad_w = max(0, need - w0)
    if pad_h or pad_w:
        tensors = [
            F.pad(t[None], (0, pad_w, 0, pad_h), mode='reflect')[0]
            for t in tensors]
    _, h, w = tensors[0].shape

    align = window_grid_alignment(window_size) if align_valid_to_window else 1
    ws = max(1, int(window_size))

    # Outer crop must fit: [0, h - (crop+2*halo)]
    max_outer_top = h - (crop + 2 * halo)
    max_outer_left = w - (crop + 2 * halo)
    if max_outer_top < 0 or max_outer_left < 0:
        raise RuntimeError('image too small for crop+halo')

    if align_valid_to_window and align > 1:
        top_candidates = list(range(0, max_outer_top + 1, align))
        left_candidates = list(range(0, max_outer_left + 1, align))
        if not top_candidates or not left_candidates:
            # Cannot satisfy full multi-scale align; fall back to any outer
            outer_top = int(rng.randint(0, max_outer_top + 1))
            outer_left = int(rng.randint(0, max_outer_left + 1))
            aligned = False
        else:
            outer_top = int(top_candidates[int(rng.randint(0, len(top_candidates)))])
            outer_left = int(left_candidates[int(rng.randint(0, len(left_candidates)))])
            aligned = True
    else:
        outer_top = int(rng.randint(0, max_outer_top + 1))
        outer_left = int(rng.randint(0, max_outer_left + 1))
        aligned = False

    valid_top = outer_top + halo
    valid_left = outer_left + halo
    y1 = outer_top + crop + 2 * halo
    x1 = outer_left + crop + 2 * halo
    out_halo = [t[:, outer_top:y1, outer_left:x1] for t in tensors]

    flip_w = bool(rng.rand() < 0.5)
    flip_h = bool(rng.rand() < 0.5)
    rot_k = int(rng.randint(0, 4)) if rng.rand() < 0.5 else 0
    if flip_w:
        out_halo = [torch.flip(t, dims=[-1]) for t in out_halo]
    if flip_h:
        out_halo = [torch.flip(t, dims=[-2]) for t in out_halo]
    if rot_k:
        out_halo = [torch.rot90(t, rot_k, dims=[-2, -1]) for t in out_halo]

    out_valid = [t[:, halo:halo + crop, halo:halo + crop] for t in out_halo]
    meta = dict(
        top=outer_top, left=outer_left,
        outer_top=outer_top, outer_left=outer_left,
        valid_top=valid_top, valid_left=valid_left,
        valid_aligned=bool(aligned),
        outer_aligned=bool(aligned),
        alignment=int(align),
        halo=halo, crop=crop,
        flip_w=flip_w, flip_h=flip_h, rot_k=rot_k,
        window_size=ws, pad_h=pad_h, pad_w=pad_w,
        valid_hw=(crop, crop),
        halo_hw=tuple(out_halo[0].shape[-2:]),
    )
    return dict(tensors_halo=out_halo, tensors_valid=out_valid, meta=meta)


def center_crop(t: torch.Tensor, halo: int, crop: Optional[int] = None) -> torch.Tensor:
    """Remove halo border; optionally force crop size."""
    if halo <= 0:
        return t if crop is None else t[..., :crop, :crop]
    if t.dim() == 3:
        h, w = t.shape[-2:]
        if crop is None:
            return t[:, halo:h - halo, halo:w - halo]
        return t[:, halo:halo + crop, halo:halo + crop]
    h, w = t.shape[-2:]
    if crop is None:
        return t[..., halo:h - halo, halo:w - halo]
    return t[..., halo:halo + crop, halo:halo + crop]
