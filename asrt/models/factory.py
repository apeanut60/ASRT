"""Unified model factory / checkpoint loader for Stage A/B/v3/v4."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from asrt.models.asrt import ASRT, build_asrt
from asrt.models.asrt_v4 import ASRTv4, build_asrt_v4


def is_v4_cfg(cfg: Optional[dict] = None, model_cfg: Optional[dict] = None) -> bool:
    cfg = dict(cfg or {})
    model_cfg = dict(model_cfg or cfg.get('model') or {})
    if bool(model_cfg.get('enable_v4', False)):
        return True
    ver = str(model_cfg.get('version', '')).lower()
    if ver in ('v4', 'asrt_v4', 'asrt-v4'):
        return True
    stage = str(cfg.get('stage', '')).lower()
    return stage in ('b_v4', 'stage_b_v4', 'v4')


def detect_ckpt_version(state: dict, ckpt_meta: Optional[dict] = None) -> str:
    """Infer model family from state_dict keys / cfg blob."""
    if ckpt_meta and isinstance(ckpt_meta, dict):
        cfg = ckpt_meta.get('cfg') or {}
        model_cfg = dict(cfg.get('model') or {})
        if is_v4_cfg(cfg, model_cfg):
            return 'v4'
        if bool(model_cfg.get('enable_v3', False)):
            return 'v3'
    keys = list(state.keys())
    if any(k.startswith('ref_encoder.') for k in keys) or any(
            k.startswith('matcher.') for k in keys):
        return 'v4'
    if any(k.startswith('reliability.') for k in keys) or any(
            'coarse_illum_adapter' in k for k in keys):
        return 'v3'
    return 'legacy'


def build_model(cfg: Optional[dict] = None, model_cfg: Optional[dict] = None):
    """Build ASRT / ASRTv3 / ASRTv4 from config dicts."""
    cfg = dict(cfg or {})
    model_cfg = dict(model_cfg or cfg.get('model') or {})
    if cfg.get('query_alignment') is not None and 'query_align' not in model_cfg:
        model_cfg['query_align'] = dict(cfg['query_alignment'])
    if is_v4_cfg(cfg, model_cfg):
        return build_asrt_v4(model_cfg)
    return build_asrt(model_cfg)


def load_checkpoint_strict(
    model: torch.nn.Module,
    ckpt_path: str,
    device: str = 'cpu',
    expect_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Load checkpoint with strict=True. Returns load report."""
    ck = torch.load(ckpt_path, map_location=device)
    state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    if not isinstance(state, dict):
        raise ValueError('checkpoint does not contain a state_dict')

    detected = detect_ckpt_version(state, ck if isinstance(ck, dict) else None)
    if expect_version is not None and detected != expect_version:
        raise RuntimeError(
            'checkpoint version mismatch: expected=%s detected=%s path=%s'
            % (expect_version, detected, ckpt_path))

    # Model family vs ckpt family
    model_is_v4 = bool(getattr(model, 'is_v4', False))
    if detected == 'v4' and not model_is_v4:
        raise RuntimeError(
            'refusing to load v4 checkpoint into non-v4 model (would silent-fail)')
    if model_is_v4 and detected != 'v4':
        raise RuntimeError(
            'refusing to load non-v4 checkpoint into ASRTv4 with strict eval loader')

    missing, unexpected = model.load_state_dict(state, strict=True)
    step = int(ck.get('step', -1)) if isinstance(ck, dict) else -1
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = dict(
        path=ckpt_path,
        version=detected,
        step=step,
        params=n_params,
        trainable=n_train,
        missing=list(missing),
        unexpected=list(unexpected),
    )
    return report


def build_and_load(
    cfg: Optional[dict],
    ckpt_path: str,
    device: str = 'cpu',
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Build model from cfg (or ckpt cfg), then strict-load weights."""
    cfg = dict(cfg or {})
    ck = torch.load(ckpt_path, map_location='cpu')
    if not cfg and isinstance(ck, dict) and isinstance(ck.get('cfg'), dict):
        cfg = dict(ck['cfg'])
    model = build_model(cfg).to(device)
    expect = 'v4' if getattr(model, 'is_v4', False) else None
    # For legacy/v3 allow detected auto; still strict load
    if expect is None:
        state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        detected = detect_ckpt_version(state, ck if isinstance(ck, dict) else None)
        if detected == 'v4':
            # cfg didn't say v4 but ckpt is v4 — rebuild
            model = build_asrt_v4(dict(cfg.get('model') or {})).to(device)
            expect = 'v4'
    report = load_checkpoint_strict(model, ckpt_path, device=device, expect_version=expect)
    model.eval()
    return model, report
