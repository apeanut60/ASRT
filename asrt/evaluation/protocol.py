"""Unified evaluation / inference protocol for Trainer and scripts/evaluate.py."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from asrt.evaluation.inference import infer_full, infer_tiled, is_irp_model


def resolve_eval_cfg(cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Merge top-level + evaluation: block into a canonical inference config.

    Defaults preserve historical Trainer auto behavior:
      inference_mode=auto, full_max_side=512, tile=256, overlap=32

    IRP formal PSNR defaults to full-image unless
    evaluation.irp_allow_tiled=true (experimental; not bit-exact vs full).
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
        irp_allow_tiled=bool(ev.get('irp_allow_tiled', False)),
        irp_oom_policy=str(ev.get('irp_oom_policy', 'error')).lower(),
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

    # Formal IRP eval: force full-image unless explicitly opted into experimental tiled.
    if is_irp_model(model) and use_tile and not ev['irp_allow_tiled']:
        use_tile = False
        # If image is huge and user asked for tiled without allow flag, fail clearly
        # when they set inference_mode=tiled (not auto).
        if mode == 'tiled':
            raise RuntimeError(
                'IRP formal evaluation refuses unverified tiled inference. '
                'Use evaluation.inference_mode=full, or set '
                'evaluation.irp_allow_tiled=true for experimental tiled '
                '(Full/Tiled are not claimed bit-exact).')

    try:
        if use_tile:
            pred = infer_tiled(
                model, low, ref,
                tile=int(ev['tile']), overlap=int(ev['overlap']))
            used = 'tiled'
        else:
            pred = infer_full(model, low, ref)
            used = 'full'
    except RuntimeError as e:
        if is_irp_model(model) and 'out of memory' in str(e).lower():
            if ev['irp_oom_policy'] == 'error':
                raise RuntimeError(
                    'IRP full-image inference OOM; refusing silent tiled fallback. '
                    'Reduce batch or enable experimental irp_allow_tiled after '
                    'reviewing Full/Tiled error stats.') from e
        raise
    meta = dict(
        inference_mode=mode,
        inference_used=used,
        tile=int(ev['tile']),
        overlap=int(ev['overlap']),
        full_max_side=int(ev['full_max_side']),
        input_hw=(H, W),
        irp_allow_tiled=bool(ev['irp_allow_tiled']),
    )
    return pred, meta
