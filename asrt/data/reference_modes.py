"""Unified Reference Mode switching for IRP (Correct/Wrong/Shuffle/Ref=Low/NoRef).

Wrong and Shuffle only change the Reference Prior Branch input.
Illumination Prior always comes from the original Low.
"""
from __future__ import annotations

from typing import List, Optional

import torch


REF_MODES = ('correct', 'wrong', 'shuffle', 'ref_low', 'noref')


def spatial_shuffle_ref(ref: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Permute spatial locations independently of content matching."""
    b, c, h, w = ref.shape
    flat = ref.view(b, c, -1)
    idx = torch.randperm(h * w, device=ref.device, generator=generator)
    return flat[:, :, idx].view(b, c, h, w)


def apply_reference_mode(
    low: torch.Tensor,
    ref: Optional[torch.Tensor],
    mode: str = 'correct',
    wrong_pool: Optional[List[torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Return the reference tensor fed to the Reference Prior Encoder.

    Modes:
      correct  — paired Nano / provided ref
      wrong    — another sample's ref from wrong_pool (or batch roll if pool empty)
      shuffle  — spatial shuffle of correct ref
      ref_low  — use Low as structural source
      noref    — None (Reference bypass)
    """
    mode = str(mode).lower().replace('-', '_')
    if mode in ('none', 'no_ref'):
        mode = 'noref'
    if mode not in REF_MODES:
        raise ValueError('unknown reference mode: %s' % mode)

    if mode == 'noref':
        return None
    if mode == 'ref_low':
        return low
    if ref is None:
        return None
    if mode == 'correct':
        return ref
    if mode == 'shuffle':
        return spatial_shuffle_ref(ref, generator=generator)
    if mode == 'wrong':
        if wrong_pool and len(wrong_pool) > 0:
            # pick a different tensor if possible
            if generator is None:
                j = int(torch.randint(0, len(wrong_pool), (1,)).item())
            else:
                j = int(torch.randint(0, len(wrong_pool), (1,), generator=generator).item())
            cand = wrong_pool[j]
            if cand.shape[-2:] != ref.shape[-2:]:
                cand = torch.nn.functional.interpolate(
                    cand if cand.dim() == 4 else cand[None],
                    size=ref.shape[-2:], mode='bilinear', align_corners=False)
                if cand.shape[0] != ref.shape[0]:
                    cand = cand.expand(ref.shape[0], -1, -1, -1)
            return cand.to(device=ref.device, dtype=ref.dtype)
        # batch roll fallback
        if ref.shape[0] > 1:
            return torch.roll(ref, shifts=1, dims=0)
        return torch.flip(ref, dims=[-1])
    return ref
