"""Halo-aware synchronized crop for IRP training.

Valid-region top/left are aligned to the full-image window grid:
  (valid_top % window_size) == 0
  (valid_left % window_size) == 0

Outer crop starts at (valid_top - halo, valid_left - halo) so HP/Sobel at the
center match full-image computation.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

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
    """Halo (pixels) for prior ops. Window alignment uses valid_* coords, not halo."""
    base = prior_context_halo(sigma=sigma, sobel=sobel)
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

    Meta includes:
      valid_top / valid_left — start of the loss region in the (padded) image
      outer_top / outer_left — start of the halo-inclusive crop
      top/left — aliases of outer_* (legacy)
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

    # Valid region must fit with halo margins on all sides.
    max_valid_top = h - crop - halo
    max_valid_left = w - crop - halo
    min_valid_top = halo
    min_valid_left = halo
    if max_valid_top < min_valid_top or max_valid_left < min_valid_left:
        raise RuntimeError('image too small for crop+halo')

    ws = max(1, int(window_size))
    if align_valid_to_window and ws > 1:
        # valid_top ≡ 0 (mod ws) relative to image origin (after pad).
        top_candidates = list(range(
            ((min_valid_top + ws - 1) // ws) * ws,
            max_valid_top + 1,
            ws,
        ))
        left_candidates = list(range(
            ((min_valid_left + ws - 1) // ws) * ws,
            max_valid_left + 1,
            ws,
        ))
        if not top_candidates or not left_candidates:
            # Fall back: clamp nearest aligned positions inside range
            vt = max(min_valid_top, min(max_valid_top, (min_valid_top // ws) * ws))
            vl = max(min_valid_left, min(max_valid_left, (min_valid_left // ws) * ws))
            # If still misaligned, disable strict align for this sample
            if vt % ws != 0 or vt < min_valid_top or vt > max_valid_top:
                vt = int(rng.randint(min_valid_top, max_valid_top + 1))
            if vl % ws != 0 or vl < min_valid_left or vl > max_valid_left:
                vl = int(rng.randint(min_valid_left, max_valid_left + 1))
            valid_top, valid_left = int(vt), int(vl)
            aligned = (valid_top % ws == 0) and (valid_left % ws == 0)
        else:
            valid_top = int(top_candidates[int(rng.randint(0, len(top_candidates)))])
            valid_left = int(left_candidates[int(rng.randint(0, len(left_candidates)))])
            aligned = True
    else:
        valid_top = int(rng.randint(min_valid_top, max_valid_top + 1))
        valid_left = int(rng.randint(min_valid_left, max_valid_left + 1))
        aligned = False

    outer_top = valid_top - halo
    outer_left = valid_left - halo
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
        # legacy aliases: outer crop origin
        top=outer_top, left=outer_left,
        outer_top=outer_top, outer_left=outer_left,
        valid_top=valid_top, valid_left=valid_left,
        valid_aligned=bool(aligned and align_valid_to_window),
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
