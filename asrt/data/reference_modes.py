"""Unified Reference Mode switching for IRP diagnostics.

Wrong / Patch-Shuffle / Pixel-Shuffle only change the Reference Prior input.
Illumination Prior always comes from the original Low.

Formal training uses Correct only; interventions are eval diagnostics.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch


REF_MODES = (
    'correct',
    'wrong',
    'patch_shuffle',
    'pixel_shuffle',
    'shuffle',  # alias → pixel_shuffle (legacy name)
    'ref_low',
    'noref',
)


def spatial_pixel_shuffle(
    ref: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Permute every spatial location (destructive / strong ablation)."""
    b, c, h, w = ref.shape
    flat = ref.view(b, c, -1)
    idx = torch.randperm(h * w, device=ref.device, generator=generator)
    return flat[:, :, idx].view(b, c, h, w)


def spatial_patch_shuffle(
    ref: torch.Tensor,
    patch_size: int = 32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Permute non-overlapping spatial patches (spatial mismatch probe)."""
    b, c, h, w = ref.shape
    ps = max(1, int(patch_size))
    # crop to multiple of patch
    hc, wc = (h // ps) * ps, (w // ps) * ps
    if hc < ps or wc < ps:
        # fall back to pixel shuffle on tiny maps
        return spatial_pixel_shuffle(ref, generator=generator)
    core = ref[:, :, :hc, :wc]
    n_h, n_w = hc // ps, wc // ps
    patches = core.view(b, c, n_h, ps, n_w, ps).permute(0, 2, 4, 1, 3, 5)
    patches = patches.contiguous().view(b, n_h * n_w, c, ps, ps)
    n = n_h * n_w
    idx = torch.randperm(n, device=ref.device, generator=generator)
    patches = patches[:, idx]
    patches = patches.view(b, n_h, n_w, c, ps, ps).permute(0, 3, 1, 4, 2, 5)
    out_core = patches.contiguous().view(b, c, hc, wc)
    if hc == h and wc == w:
        return out_core
    out = ref.clone()
    out[:, :, :hc, :wc] = out_core
    return out


# legacy alias
spatial_shuffle_ref = spatial_pixel_shuffle


def load_wrong_mapping(path: str) -> Dict[str, str]:
    with open(path, 'r') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError('wrong mapping must be a JSON object sample_id→sample_id')
    mapping = {str(k): str(v) for k, v in raw.items()}
    for k, v in mapping.items():
        if k == v:
            raise ValueError('wrong mapping self-pair forbidden: %s' % k)
    return mapping


def build_wrong_mapping(
    sample_ids: List[str],
    seed: int = 42,
    out_path: Optional[str] = None,
) -> Dict[str, str]:
    """Fixed Wrong-scene mapping: each id → a different id (derangement-ish).

    Uses a seeded RNG. Does not attempt near-duplicate filtering beyond
    forbidding self-pairs (manifest metadata for near-dup may be unavailable).
    """
    ids = [str(s) for s in sample_ids]
    if len(ids) < 2:
        raise ValueError('need >=2 samples to build wrong reference mapping')
    rng = np.random.RandomState(int(seed))
    # Produce a derangement via repeated random permutation
    mapping = None
    for _ in range(1000):
        perm = rng.permutation(len(ids))
        if all(perm[i] != i for i in range(len(ids))):
            mapping = {ids[i]: ids[int(perm[i])] for i in range(len(ids))}
            break
    if mapping is None:
        # Fallback cycle shift
        mapping = {ids[i]: ids[(i + 1) % len(ids)] for i in range(len(ids))}
        if len(ids) == 1:
            raise ValueError('cannot build wrong mapping for single sample')
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(mapping, f, indent=2, sort_keys=True)
    return mapping


def resolve_wrong_ref_id(
    sample_id: str,
    mapping: Dict[str, str],
) -> str:
    sid = str(sample_id)
    if sid not in mapping:
        raise KeyError(
            'sample_id %r missing from wrong-reference mapping (no flip fallback)'
            % sid)
    wid = str(mapping[sid])
    if wid == sid:
        raise ValueError('wrong mapping points to self for %r' % sid)
    return wid


def apply_reference_mode(
    low: torch.Tensor,
    ref: Optional[torch.Tensor],
    mode: str = 'correct',
    *,
    wrong_ref: Optional[torch.Tensor] = None,
    wrong_pool: Optional[List[torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    patch_size: int = 32,
) -> Optional[torch.Tensor]:
    """Return the reference tensor fed to the Reference Prior Encoder.

    Modes:
      correct       — paired Nano / provided ref
      wrong         — requires wrong_ref (from fixed mapping); no flip fallback
      patch_shuffle — block permutation of correct ref
      pixel_shuffle — per-pixel permutation (also accepts legacy 'shuffle')
      ref_low       — use Low as structural source
      noref         — None (Reference bypass)
    """
    mode = str(mode).lower().replace('-', '_')
    if mode in ('none', 'no_ref'):
        mode = 'noref'
    if mode == 'shuffle':
        mode = 'pixel_shuffle'
    if mode not in REF_MODES:
        raise ValueError('unknown reference mode: %s' % mode)

    if mode == 'noref':
        return None
    if mode == 'ref_low':
        return low
    if ref is None and mode in ('correct', 'patch_shuffle', 'pixel_shuffle'):
        return None
    if mode == 'correct':
        return ref
    if mode == 'pixel_shuffle':
        return spatial_pixel_shuffle(ref, generator=generator)
    if mode == 'patch_shuffle':
        return spatial_patch_shuffle(ref, patch_size=patch_size, generator=generator)
    if mode == 'wrong':
        if wrong_ref is not None:
            wr = wrong_ref
            if wr.dim() == 3:
                wr = wr.unsqueeze(0)
            if tuple(wr.shape[-2:]) != tuple(ref.shape[-2:] if ref is not None else wr.shape[-2:]):
                target = ref.shape[-2:] if ref is not None else low.shape[-2:]
                wr = torch.nn.functional.interpolate(
                    wr, size=target, mode='bilinear', align_corners=False)
            return wr.to(device=low.device, dtype=low.dtype)
        # Deprecated pool path kept only if explicitly provided AND size>1 with
        # no self; still no horizontal-flip fallback.
        if wrong_pool and len(wrong_pool) > 0:
            if generator is None:
                j = int(torch.randint(0, len(wrong_pool), (1,)).item())
            else:
                j = int(torch.randint(
                    0, len(wrong_pool), (1,), generator=generator).item())
            cand = wrong_pool[j]
            if cand.dim() == 3:
                cand = cand.unsqueeze(0)
            target = ref.shape[-2:] if ref is not None else low.shape[-2:]
            if tuple(cand.shape[-2:]) != tuple(target):
                cand = torch.nn.functional.interpolate(
                    cand, size=target, mode='bilinear', align_corners=False)
            return cand.to(device=low.device, dtype=low.dtype)
        raise RuntimeError(
            'wrong reference mode requires a fixed mapping / wrong_ref tensor; '
            'horizontal-flip and silent self-substitution are forbidden')
    return ref
