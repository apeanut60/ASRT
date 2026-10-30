"""ASRT-v4: geometry-constrained soft matching + dual reliability, single backbone."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asrt.models.csfi_unet import CSFIUNet, resize_like
from asrt.models.v4_adapters import LowPassLuminance
from asrt.models.v4_matching import LocalSoftMatcher, soft_aggregate_scalar, warp_by_disp
from asrt.models.v4_ref_encoder import RefEncoderV4
from asrt.models.v4_reliability import (
    DualReliabilityEstimator,
    build_reliability_features,
    upsample_gate,
)


def _gate_stats(g: torch.Tensor) -> dict:
    flat = g.detach().float().reshape(-1)
    return dict(
        mean=float(flat.mean()),
        std=float(flat.std(unbiased=False)),
        p10=float(torch.quantile(flat, 0.10)),
        p50=float(torch.quantile(flat, 0.50)),
        p90=float(torch.quantile(flat, 0.90)),
        frac_near0=float((flat < 0.05).float().mean()),
        frac_near1=float((flat > 0.95).float().mean()),
    )


class ASRTv4(nn.Module):
    """Single CSFI-U-Net restore backbone + trainable local soft matching.

    forward(low, ref=None) → enhanced in [-1,1]
    No Frozen LTE / SearchTransfer / Hard Argmax.
    """

    def __init__(
        self,
        num_res_blocks='8+8+4+2',
        n_feats=64,
        res_scale=1.,
        igab_blocks=(1, 1),
        igab_heads=1,
        enable_lightup=True,
        match_dim: int = 64,
        match_radius: int = 4,
        match_temperature: float = 0.1,
        match_lambda_pos: float = 0.5,
        ref_base_ch: int = 32,
        reli_mid: int = 64,
        d_clamp: float = 2.0,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.is_v4 = True
        self.enable_v3 = False
        self.n_feats = int(n_feats)
        self.match_dim = int(match_dim)
        if self.match_dim != self.n_feats:
            raise ValueError(
                'Current ASRT-v4 requires match_dim == n_feats '
                '(got match_dim=%d, n_feats=%d). Separate Matching Q/K vs '
                'restoration feature dims are not supported without explicit '
                'value-projection modules.' % (self.match_dim, self.n_feats))
        self.d_clamp = float(d_clamp)
        self.eps = float(eps)
        # Stash non-weight hyperparams for checkpoint consistency checks
        self.v4_hparams = dict(
            version='v4',
            n_feats=self.n_feats,
            num_res_blocks=num_res_blocks if not isinstance(num_res_blocks, list)
            else '+'.join(str(x) for x in num_res_blocks),
            enable_lightup=bool(enable_lightup),
            match_dim=self.match_dim,
            match_radius=int(match_radius),
            match_temperature=float(match_temperature),
            match_lambda_pos=float(match_lambda_pos),
            ref_base_ch=int(ref_base_ch),
            reli_mid=int(reli_mid),
        )
        self.backbone = CSFIUNet(
            num_res_blocks=num_res_blocks,
            n_feats=n_feats,
            res_scale=res_scale,
            igab_blocks=igab_blocks,
            igab_heads=igab_heads,
            enable_lightup=enable_lightup,
            use_global_illum=False,
            use_ref_illum=False,
            enable_v3=False,
            enable_v4=True,
        )
        self.ref_encoder = RefEncoderV4(base_ch=ref_base_ch, out_ch=match_dim)
        self.matcher = LocalSoftMatcher(
            dim=match_dim,
            radius=match_radius,
            temperature=match_temperature,
            lambda_pos=match_lambda_pos,
        )
        # reliability input channels from build_reliability_features
        self.reliability = DualReliabilityEstimator(in_ch=10, mid=reli_mid)
        self.lpf = LowPassLuminance(ksize=9, sigma=3.0)
        # Align low stem feat (n_feats) to match_dim if needed
        if n_feats != match_dim:
            self.low_q_proj = nn.Conv2d(n_feats, match_dim, 1)
            self.illum_q_proj = nn.Conv2d(n_feats, match_dim, 1)
        else:
            self.low_q_proj = nn.Identity()
            self.illum_q_proj = nn.Identity()
        # Legacy VGG inject heads are unused in v4 forward; keep for Stage-A
        # weight compatibility but freeze so they are not trained / counted active.
        self._freeze_unused_inject_heads()
        self.last_aux = None

    def _freeze_unused_inject_heads(self) -> None:
        for name in ('conv11_head', 'conv22_head', 'conv33_head'):
            mod = getattr(self.backbone, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False

    @staticmethod
    def _resize_to(ref, hw):
        if tuple(ref.shape[-2:]) == tuple(hw):
            return ref
        return F.interpolate(ref, size=tuple(hw), mode='bilinear', align_corners=True)

    def _build_illum_prior(
        self,
        ref_m11: torch.Tensor,
        base_m11: torch.Tensor,
        weight: torch.Tensor,
        disp: torch.Tensor,
    ) -> dict:
        """Soft-aggregate P_ref at H/4 from [0,1] luminance proxies."""
        # Explicit [-1,1] → [0,1] before LPF (P0 numerical-domain fix)
        ref01 = ((ref_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
        base01 = ((base_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
        p_ref = self.lpf(ref01)
        p_base = self.lpf(base01)
        h4, w4 = weight.shape[-2:]
        p_ref_h4 = F.interpolate(p_ref, size=(h4, w4), mode='bilinear', align_corners=True)
        p_base_h4 = F.interpolate(p_base, size=(h4, w4), mode='bilinear', align_corners=True)
        p_ref_m = soft_aggregate_scalar(p_ref_h4, weight, self.matcher.radius)
        d = (torch.log(p_ref_m.clamp_min(self.eps))
             - torch.log(p_base_h4.clamp_min(self.eps))).clamp(-self.d_clamp, self.d_clamp)
        return dict(
            p_ref_matched=p_ref_m,
            p_base=p_base_h4,
            d_ref=d,
            p_ref_full_lpf=p_ref,
            disp=disp,
        )

    def match_reference(self, stem: dict, ref: torch.Tensor):
        """Trainable local soft matching → multiscale T + reliability + illum prior."""
        low_pad = stem['low']
        orig_hw = stem['orig_hw']
        # Resize to *original* Low spatial size, then apply the same pad as Low.
        # Do not treat pad pixels as a resize target (geometry consistency).
        ref = self._resize_to(ref, orig_hw)
        ref = CSFIUNet.safe_pad(ref, low_pad.shape[-2:])

        f1, f2, f3 = self.ref_encoder(ref)
        # Align spatial if off-by-one from stem
        f3 = resize_like(f3, stem['e2'], mode='bilinear')
        f2 = resize_like(f2, stem['e1'], mode='bilinear')
        f1 = resize_like(f1, stem['e0'], mode='bilinear')

        q_low = self.low_q_proj(stem['e2'])
        q_ill = self.illum_q_proj(stem['illu_h4'])
        t_h4, weight, diag = self.matcher(q_low, q_ill, f3)

        # Multiscale warp via expected displacement (H/2 scale=2, H scale=4)
        t_h2 = warp_by_disp(f2, diag['disp'], scale=2.0)
        t_h1 = warp_by_disp(f1, diag['disp'], scale=4.0)

        reli_in = build_reliability_features(stem['e2'], t_h4, diag)
        c_t, c_l = self.reliability(reli_in)
        illum_prior = self._build_illum_prior(ref, stem['base'], weight, diag['disp'])
        return dict(
            T_h4=t_h4, T_h2=t_h2, T_h1=t_h1,
            c_t=c_t, c_l=c_l,
            illum_prior=illum_prior,
            match_diag=diag,
            ref_aligned=ref,
        )

    def forward(self, low, ref=None, return_aux=False, query_gamma=None, query_lut=None):
        # query_gamma/lut accepted for inference API compatibility; unused in v4.
        del query_gamma, query_lut
        if low.dim() != 4 or low.size(1) != 3:
            raise ValueError('low must be [B,3,H,W]')
        use_ref = ref is not None
        stem = self.backbone.encode_low_stem(low)

        pack = None
        if use_ref:
            if ref.dim() != 4 or ref.size(1) != 3:
                raise ValueError('ref must be [B,3,H,W] or None')
            if ref.size(0) != low.size(0):
                raise ValueError('ref/low batch mismatch')
            pack = self.match_reference(stem, ref)

        out = self.backbone(
            low,
            S=None,
            T_lv3=None if pack is None else pack['T_h4'],
            T_lv2=None if pack is None else pack['T_h2'],
            T_lv1=None if pack is None else pack['T_h1'],
            ref=None if pack is None else pack['ref_aligned'],
            use_reference=use_ref,
            c_t=None if pack is None else pack['c_t'],
            c_l=None if pack is None else pack['c_l'],
            illum_prior=None if pack is None else pack['illum_prior'],
            stem=stem,
        )

        aux = dict(used_reference=use_ref, v4=self.backbone.last_v4_diag)
        if pack is not None:
            aux['c_t_stats'] = _gate_stats(pack['c_t'])
            aux['c_l_stats'] = _gate_stats(pack['c_l'])
            md = pack['match_diag']
            aux['match'] = dict(
                entropy=float(md['entropy'].detach().mean()),
                max_weight=float(md['max_weight'].detach().mean()),
                disp_mag=float(md['disp'].detach().norm(dim=1).mean()),
                disp_var=float(md['disp_var'].detach().mean()),
                continuity=float(md['continuity'].detach().mean()),
                q_norm=float(md['q_norm'].detach().mean()),
                k_norm=float(md['k_norm'].detach().mean()),
                q_norm_pre=float(md['q_norm_pre'].detach().mean()),
                k_norm_pre=float(md['k_norm_pre'].detach().mean()),
                n_valid_mean=float(md['n_valid'].detach().mean()),
            )
            # keep tensors for viz (detached)
            aux['maps'] = dict(
                c_t=pack['c_t'].detach(),
                c_l=pack['c_l'].detach(),
                disp=md['disp'].detach(),
                entropy=md['entropy'].detach(),
                p_ref=pack['illum_prior']['p_ref_matched'].detach(),
            )
        self.last_aux = aux
        if return_aux:
            return out, aux
        return out

    def count_parameters(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_asrt_v4(cfg: Optional[dict] = None) -> ASRTv4:
    cfg = dict(cfg or {})
    match = dict(cfg.get('matching', {}) or {})
    return ASRTv4(
        num_res_blocks=cfg.get('num_res_blocks', '8+8+4+2'),
        n_feats=int(cfg.get('n_feats', 64)),
        res_scale=float(cfg.get('res_scale', 1.0)),
        enable_lightup=bool(cfg.get('enable_lightup', True)),
        match_dim=int(match.get('dim', cfg.get('match_dim', 64))),
        match_radius=int(match.get('radius', cfg.get('match_radius', 4))),
        match_temperature=float(match.get('temperature', cfg.get('match_temperature', 0.1))),
        match_lambda_pos=float(match.get('lambda_pos', cfg.get('match_lambda_pos', 0.5))),
        ref_base_ch=int(cfg.get('ref_base_ch', 32)),
        reli_mid=int(cfg.get('reli_mid', 64)),
    )


def load_stage_a_into_v4(model: ASRTv4, state: dict) -> dict:
    """Load Stage A restoration backbone into v4. Report loaded / new / mismatched keys.

    Does not silently ignore missing backbone weights. LTE keys in ckpt are expected
    unused (v4 has no LTE) and listed separately.
    """
    if not isinstance(state, dict):
        raise ValueError('state must be a state_dict')
    if 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    model_sd = model.state_dict()

    loaded = {}
    shape_mismatch = []
    unused_ckpt = []  # e.g. lte.*
    for k, v in state.items():
        if k not in model_sd:
            unused_ckpt.append(k)
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        loaded[k] = v

    if shape_mismatch:
        raise RuntimeError('shape mismatch loading Stage A into v4: %s' % shape_mismatch[:15])

    # Every backbone key present in Stage A must load; also allow missing inject heads
    # only if shapes differ — they should match for Stage A.
    backbone_ckpt = [k for k in state if k.startswith('backbone.')]
    missing_backbone = [k for k in backbone_ckpt if k not in loaded]
    if missing_backbone:
        raise RuntimeError(
            'Stage A backbone keys failed to load into v4: %s' % missing_backbone[:20])

    missing, unexpected = model.load_state_dict(loaded, strict=False)
    if unexpected:
        raise RuntimeError('unexpected keys after filtered load: %s' % list(unexpected)[:20])

    # Expected new modules in v4
    new_prefixes = (
        'ref_encoder.', 'matcher.', 'reliability.', 'lpf.',
        'low_q_proj.', 'illum_q_proj.',
        'backbone.tex_adapt_', 'backbone.illum_adapt_',
    )
    bad_missing = []
    expected_new = []
    for k in missing:
        if any(k.startswith(p) or p.rstrip('.') in k for p in new_prefixes):
            expected_new.append(k)
        elif k.startswith('backbone.') and any(
                x in k for x in ('tex_adapt_', 'illum_adapt_')):
            expected_new.append(k)
        else:
            # Identity modules have no params; BN buffers etc.
            bad_missing.append(k)
    # Identity has no keys; filter true anomalies
    bad_missing = [k for k in bad_missing if 'Identity' not in k]
    if bad_missing:
        raise RuntimeError(
            'unexpected missing keys (not new v4 modules): %s' % bad_missing[:30])

    report = dict(
        loaded=len(loaded),
        expected_new=sorted(expected_new),
        unused_ckpt=sorted(unused_ckpt),
        shape_mismatch=[],
        missing_after_load=sorted(missing),
    )
    return report
