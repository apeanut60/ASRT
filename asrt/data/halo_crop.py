"""Halo-aware synchronized crop for IRP training.

Crop coordinates are chosen for the *valid* center region; tensors are extracted
with surrounding halo so Gaussian HP / Sobel at the center match full-image
computation within numerical tolerance.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from asrt.models.prior_ops import prior_context_halo


def compute_irp_halo(
    sigma: float = 7.0,
    window_size: int = 8,
    sobel: int = 1,
    extra: int = 0,
) -> int:
    """Halo (pixels) for prior ops + optional window-align pad."""
    base = prior_context_halo(sigma=sigma, sobel=sobel)
    # Align outer crop so that after removing halo, window grid of the
    # valid crop matches a global (top-left origin) window lattice when
    # top/left of the *valid* region are multiples of window_size.
    return int(base + int(extra))


def sync_geometry_halo(
    tensors: Sequence[torch.Tensor],
    crop: int,
    rng: np.random.RandomState,
    halo: int,
    window_size: int = 8,
    align_valid_to_window: bool = True,
) -> Dict[str, object]:
    """Shared random crop with halo context + flips/rotations.

    Returns dict:
      tensors_halo: list of [C,H+2h?,W+2h?] crops including halo
      tensors_valid: list of center crop (size crop×crop) after same aug
      meta: top,left,halo,flips,rot,valid_hw
    """
    assert len(tensors) >= 1
    halo = max(0, int(halo))
    crop = int(crop)
    _, h, w = tensors[0].shape
    need = crop + 2 * halo
    pad_h = max(0, need - h)
    pad_w = max(0, need - w)
    if pad_h or pad_w:
        tensors = [
            F.pad(t[None], (0, pad_w, 0, pad_h), mode='reflect')[0]
            for t in tensors]
        _, h, w = tensors[0].shape

    # Valid-region top/left inside the (possibly padded) image, leaving halo.
    max_top = h - crop - 2 * halo
    max_left = w - crop - 2 * halo
    if max_top < 0 or max_left < 0:
        raise RuntimeError('image too small for crop+halo')
    ws = max(1, int(window_size))
    if align_valid_to_window and ws > 1:
        # Snap valid top/left to window grid relative to image origin.
        top_candidates = list(range(0, max_top + 1, ws))
        left_candidates = list(range(0, max_left + 1, ws))
        if not top_candidates:
            top_candidates = [0]
        if not left_candidates:
            left_candidates = [0]
        top = int(top_candidates[int(rng.randint(0, len(top_candidates)))])
        left = int(left_candidates[int(rng.randint(0, len(left_candidates)))])
    else:
        top = int(rng.randint(0, max_top + 1))
        left = int(rng.randint(0, max_left + 1))

    y0, x0 = top, left
    y1, x1 = top + crop + 2 * halo, left + crop + 2 * halo
    out_halo = [t[:, y0:y1, x0:x1] for t in tensors]

    flip_w = bool(rng.rand() < 0.5)
    flip_h = bool(rng.rand() < 0.5)
    rot_k = int(rng.randint(0, 4)) if rng.rand() < 0.5 else 0
    if flip_w:
        out_halo = [torch.flip(t, dims=[-1]) for t in out_halo]
    if flip_h:
        out_halo = [torch.flip(t, dims=[-2]) for t in out_halo]
    if rot_k:
        out_halo = [torch.rot90(t, rot_k, dims=[-2, -1]) for t in out_halo]

    # Center valid crop
    out_valid = [t[:, halo:halo + crop, halo:halo + crop] for t in out_halo]
    meta = dict(
        top=top, left=left, halo=halo, crop=crop,
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
