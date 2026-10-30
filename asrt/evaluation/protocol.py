"""Unified evaluation / inference protocol for Trainer and scripts/evaluate.py."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from asrt.evaluation.inference import infer_full, infer_tiled


def resolve_eval_cfg(cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Merge top-level + evaluation: block into a canonical inference config.

    Defaults preserve historical Trainer auto behavior:
      inference_mode=auto, full_max_side=512, tile=256, overlap=32
    """
    cfg = dict(cfg or {})
    ev = dict(cfg.get('evaluation') or {})
    mode = str(ev.get(
        'inference_mode',
        cfg.get('eval_inference_mode', 'auto'))).lower()
    if mode not in ('auto', 'full', 'tiled'):
        raise ValueError('evaluation.inference_mode must be auto|full|tiled, got %r' % mode)
    return dict(
        inference_mode=mode,
        full_max_side=int(ev.get(
            'full_max_side',
            cfg.get('eval_full_max_side', 512))),
        tile=int(ev.get('tile', cfg.get('eval_tile', 256))),
        overlap=int(ev.get('overlap', cfg.get('eval_overlap', 32))),
        input_range=str(ev.get('input_range', 'minus_one_one')),
    )


@torch.no_grad()
def infer_with_protocol(model, low, ref=None, eval_cfg: Optional[dict] = None):
    """Run Full or Tiled inference according to unified protocol."""
    ev = resolve_eval_cfg(eval_cfg if eval_cfg is not None else {})
    mode = ev['inference_mode']
    H, W = int(low.shape[-2]), int(low.shape[-1])
    use_tile = False
    if mode == 'tiled':
        use_tile = True
    elif mode == 'full':
        use_tile = False
    else:  # auto
        use_tile = max(H, W) > int(ev['full_max_side'])
    if use_tile:
        pred = infer_tiled(
            model, low, ref,
            tile=int(ev['tile']), overlap=int(ev['overlap']))
        used = 'tiled'
    else:
        pred = infer_full(model, low, ref)
        used = 'full'
    meta = dict(
        inference_mode=mode,
        inference_used=used,
        tile=int(ev['tile']),
        overlap=int(ev['overlap']),
        full_max_side=int(ev['full_max_side']),
        input_hw=(H, W),
    )
    return pred, meta
