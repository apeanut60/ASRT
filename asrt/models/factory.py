"""Unified model factory / checkpoint loader for Stage A/B/v3/v4."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from asrt.models.asrt import ASRT, build_asrt
from asrt.models.asrt_v4 import ASRTv4, build_asrt_v4

# Non-weight hyperparams that change v4 forward results
_V4_HPARAM_KEYS = (
    'version', 'n_feats', 'num_res_blocks', 'enable_lightup',
    'match_dim', 'match_radius', 'match_temperature', 'match_lambda_pos',
    'ref_base_ch', 'reli_mid',
)


def _extract_v4_hparams(cfg: Optional[dict]) -> Dict[str, Any]:
    cfg = dict(cfg or {})
    model = dict(cfg.get('model') or {})
    match = dict(model.get('matching') or {})
    return dict(
        version=str(model.get('version', 'v4')),
        n_feats=int(model.get('n_feats', 64)),
        num_res_blocks=str(model.get('num_res_blocks', '8+8+4+2')),
        enable_lightup=bool(model.get('enable_lightup', True)),
        match_dim=int(match.get('dim', model.get('match_dim', 64))),
        match_radius=int(match.get('radius', model.get('match_radius', 4))),
        match_temperature=float(match.get(
            'temperature', model.get('match_temperature', 0.1))),
        match_lambda_pos=float(match.get(
            'lambda_pos', model.get('match_lambda_pos', 0.5))),
        ref_base_ch=int(model.get('ref_base_ch', 32)),
        reli_mid=int(model.get('reli_mid', 64)),
    )


def compare_v4_hparams(a: dict, b: dict) -> List[dict]:
    conflicts = []
    for k in _V4_HPARAM_KEYS:
        if k not in a or k not in b:
            continue
        va, vb = a[k], b[k]
        if isinstance(va, float) or isinstance(vb, float):
            if abs(float(va) - float(vb)) > 1e-8:
                conflicts.append(dict(key=k, expected=va, got=vb))
        else:
            if va != vb:
                conflicts.append(dict(key=k, expected=va, got=vb))
    return conflicts


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
    external_cfg: Optional[dict] = None,
    allow_hparam_override: bool = False,
) -> Dict[str, Any]:
    """Load checkpoint with strict=True. Returns load report.

    For v4, also compares non-weight hyperparams in ckpt cfg vs external_cfg /
    model.v4_hparams. Conflicts raise unless allow_hparam_override=True.
    Legacy checkpoints without full cfg get a compatibility note (no hard fail).
    """
    ck = torch.load(ckpt_path, map_location=device)
    state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    if not isinstance(state, dict):
        raise ValueError('checkpoint does not contain a state_dict')

    detected = detect_ckpt_version(state, ck if isinstance(ck, dict) else None)
    if expect_version is not None and detected != expect_version:
        raise RuntimeError(
            'checkpoint version mismatch: expected=%s detected=%s path=%s'
            % (expect_version, detected, ckpt_path))

    model_is_v4 = bool(getattr(model, 'is_v4', False))
    if detected == 'v4' and not model_is_v4:
        raise RuntimeError(
            'refusing to load v4 checkpoint into non-v4 model (would silent-fail)')
    if model_is_v4 and detected != 'v4':
        raise RuntimeError(
            'refusing to load non-v4 checkpoint into ASRTv4 with strict eval loader')

    hparam_conflicts: List[dict] = []
    hparam_note = None
    ckpt_cfg = ck.get('cfg') if isinstance(ck, dict) else None
    if model_is_v4:
        model_hp = dict(getattr(model, 'v4_hparams', {}) or {})
        if ckpt_cfg and isinstance(ckpt_cfg, dict) and ckpt_cfg.get('model'):
            ckpt_hp = _extract_v4_hparams(ckpt_cfg)
            hparam_conflicts = compare_v4_hparams(ckpt_hp, model_hp)
            if external_cfg:
                ext_hp = _extract_v4_hparams(external_cfg)
                hparam_conflicts += compare_v4_hparams(ckpt_hp, ext_hp)
        else:
            hparam_note = (
                'checkpoint lacks full v4 cfg; skipped hyperparam consistency check '
                '(legacy-compatible). Weights still loaded with strict=True.')
        if hparam_conflicts and not allow_hparam_override:
            raise RuntimeError(
                'v4 hyperparam mismatch vs checkpoint (pass allow_hparam_override '
                'only for diagnostics): %s' % hparam_conflicts)
    elif detected == 'legacy' and external_cfg and not (
            isinstance(ck, dict) and isinstance(ck.get('cfg'), dict)):
        hparam_note = 'legacy checkpoint; hyperparam check not enforced'

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
        hparam_conflicts=hparam_conflicts,
        hparam_override=bool(allow_hparam_override and hparam_conflicts),
        hparam_note=hparam_note,
    )
    return report


def build_and_load(
    cfg: Optional[dict],
    ckpt_path: str,
    device: str = 'cpu',
    allow_hparam_override: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Build model from cfg (or ckpt cfg), then strict-load weights."""
    cfg = dict(cfg or {})
    ck = torch.load(ckpt_path, map_location='cpu')
    if not cfg and isinstance(ck, dict) and isinstance(ck.get('cfg'), dict):
        cfg = dict(ck['cfg'])
    # Prefer checkpoint cfg for v4 construction when both present and no override
    if (isinstance(ck, dict) and isinstance(ck.get('cfg'), dict)
            and is_v4_cfg(ck['cfg']) and not allow_hparam_override):
        if cfg and is_v4_cfg(cfg):
            conflicts = compare_v4_hparams(
                _extract_v4_hparams(ck['cfg']), _extract_v4_hparams(cfg))
            if conflicts:
                raise RuntimeError(
                    'external config conflicts with checkpoint v4 hparams: %s'
                    % conflicts)
        cfg = dict(ck['cfg'])
    model = build_model(cfg).to(device)
    expect = 'v4' if getattr(model, 'is_v4', False) else None
    if expect is None:
        state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        detected = detect_ckpt_version(state, ck if isinstance(ck, dict) else None)
        if detected == 'v4':
            model = build_asrt_v4(dict(cfg.get('model') or {})).to(device)
            expect = 'v4'
    report = load_checkpoint_strict(
        model, ckpt_path, device=device, expect_version=expect,
        external_cfg=cfg, allow_hparam_override=allow_hparam_override)
    model.eval()
    return model, report
