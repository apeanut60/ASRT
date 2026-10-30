"""Top-level ASRT model: dual-input with true no-reference bypass (+ optional v3)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from asrt.models.csfi_unet import CSFIUNet, parse_num_res_blocks
from asrt.models.illumination_prior import build_matched_illumination
from asrt.models.query_illum_align import (
    align_query_illumination,
    estimate_query_align_maps,
)
from asrt.models.reference import LTE, SearchTransfer
from asrt.models.reliability import DualReliabilityEstimator, reliability_stats


def _default_query_align_cfg():
    return dict(
        enabled=False,
        method='gamma_hist',
        gamma_alpha=0.75,
        hist_beta=0.50,  # verified working recipe in TTSR-lowlight-v1-data1SOTA (not the 0.20 fallback)
        gamma_min=0.3,
        gamma_max=1.0,
        hist_bins=256,
        stats_scope='full_image',
    )


class ASRT(nn.Module):
    """Illumination-Guided Cross-Scale Reference Restoration.

    forward(low, ref=None, return_aux=False) -> enhanced [-1,1]
    When ref is None, LTE/SearchTransfer/ref inject/RefIllum/v3 are skipped.
    Query alignment (optional) only affects LTE Query; backbone always gets Raw Low.
    """

    def __init__(
        self,
        num_res_blocks='8+8+4+2',
        n_feats=64,
        res_scale=1.,
        igab_blocks=(1, 1),
        igab_heads=1,
        freeze_lte=True,
        use_global_illum=False,
        use_ref_illum=False,
        enable_lightup=True,
        query_align: Optional[dict] = None,
        enable_v3: bool = False,
    ):
        super().__init__()
        self.enable_v3 = bool(enable_v3)
        self.backbone = CSFIUNet(
            num_res_blocks=num_res_blocks,
            n_feats=n_feats,
            res_scale=res_scale,
            igab_blocks=igab_blocks,
            igab_heads=igab_heads,
            enable_lightup=enable_lightup,
            use_global_illum=use_global_illum,
            use_ref_illum=use_ref_illum,
            enable_v3=self.enable_v3,
        )
        self.lte = LTE(requires_grad=not freeze_lte)
        self.search_transfer = SearchTransfer()
        self.freeze_lte = bool(freeze_lte)
        qa = _default_query_align_cfg()
        if query_align:
            qa.update(dict(query_align))
        self.query_align_cfg = qa
        self.reliability = DualReliabilityEstimator() if self.enable_v3 else None
        self.last_aux = None
        self.last_query_illum_info = None

    @staticmethod
    def _resize_to(ref, hw):
        if tuple(ref.shape[-2:]) == tuple(hw):
            return ref
        return F.interpolate(ref, size=tuple(hw), mode='bilinear', align_corners=False)

    def _align_query(self, low, ref, gamma_override=None, lut_override=None):
        cfg = self.query_align_cfg
        if not cfg.get('enabled', False) or ref is None:
            return low, None
        method = str(cfg.get('method', 'gamma_hist')).lower()
        alpha = float(cfg.get('gamma_alpha', 0.75))
        beta = float(cfg.get('hist_beta', 0.0))
        if method in ('none', 'off', 'raw'):
            return low, None
        if method == 'gamma':
            beta = 0.0
        elif method == 'hist':
            alpha = 0.0
        # gamma_hist / default: both
        q, info = align_query_illumination(
            low, ref,
            alpha=alpha,
            beta=beta,
            gamma_min=float(cfg.get('gamma_min', 0.3)),
            gamma_max=float(cfg.get('gamma_max', 1.0)),
            hist_bins=int(cfg.get('hist_bins', 256)),
            gamma_override=gamma_override,
            lut_override=lut_override,
        )
        return q, info

    def match_reference(self, low, ref, gamma_override=None, lut_override=None):
        """Run LTE + SearchTransfer. Inputs in [-1, 1].

        Query may be illumination-aligned; Key/Value always from raw Nano Ref.
        Backbone input (Raw Low) is NOT modified here.
        """
        ref = self._resize_to(ref, low.shape[-2:])
        import math
        th = int(math.ceil(low.shape[-2] / 4.0) * 4)
        tw = int(math.ceil(low.shape[-1] / 4.0) * 4)
        low_p = CSFIUNet.safe_pad(low, (th, tw))
        ref_p = CSFIUNet.safe_pad(ref, (th, tw))

        low_q, qinfo = self._align_query(
            low_p, ref_p, gamma_override=gamma_override, lut_override=lut_override)
        self.last_query_illum_info = qinfo

        q01 = (low_q + 1.) * 0.5
        r01 = (ref_p + 1.) * 0.5
        _, _, q_lv3 = self.lte(q01)
        ref_lv1, ref_lv2, ref_lv3 = self.lte(r01)
        _, _, k_lv3 = self.lte(r01)
        if self.enable_v3:
            S, T3, T2, T1, diag = self.search_transfer(
                q_lv3, k_lv3, ref_lv1, ref_lv2, ref_lv3, return_diag=True)
        else:
            S, T3, T2, T1 = self.search_transfer(
                q_lv3, k_lv3, ref_lv1, ref_lv2, ref_lv3, return_diag=False)
            diag = None
        return S, T3, T2, T1, ref_p, diag

    def forward(
        self,
        low,
        ref=None,
        return_aux=False,
        query_gamma=None,
        query_lut=None,
    ):
        if low.dim() != 4 or low.size(1) != 3:
            raise ValueError('low must be [B,3,H,W]')
        use_ref = ref is not None
        S = T3 = T2 = T1 = None
        ref_aligned = None
        match_diag = None
        c_t = c_l = None
        illum_prior = None
        self.last_query_illum_info = None
        if use_ref:
            if ref.dim() != 4 or ref.size(1) != 3:
                raise ValueError('ref must be [B,3,H,W] or None')
            if ref.size(0) != low.size(0):
                raise ValueError('ref/low batch mismatch')
            S, T3, T2, T1, ref_aligned, match_diag = self.match_reference(
                low, ref, gamma_override=query_gamma, lut_override=query_lut)

            if self.enable_v3 and match_diag is not None:
                # Pad low to match SearchTransfer grid for structure maps
                import math
                th = int(math.ceil(low.shape[-2] / 4.0) * 4)
                tw = int(math.ceil(low.shape[-1] / 4.0) * 4)
                low_p = CSFIUNet.safe_pad(low, (th, tw))
                c_t, c_l = self.reliability(match_diag, low_p)

                # Base light-up for relative illumination (Stage-A style, no GT)
                with torch.no_grad():
                    # Use current illum estimator path for P_base proxy from Raw Low
                    low01 = (low_p + 1.) * 0.5
                    _, log_gain = self.backbone.illumination_estimator(low01)
                    lightup = low01 * torch.exp(log_gain.clamp(-1.5, 1.5))
                    base_m11 = 2. * lightup - 1.
                h4, w4 = match_diag['grid_hw']
                illum_prior = build_matched_illumination(
                    ref_aligned, base_m11, match_diag['match_index'], h4, w4)

        # Backbone always receives Raw Low (never aligned query).
        out = self.backbone(
            low, S=S, T_lv3=T3, T_lv2=T2, T_lv1=T1,
            ref=ref_aligned if use_ref else None,
            use_reference=use_ref,
            c_t=c_t, c_l=c_l, illum_prior=illum_prior,
        )
        aux = dict(
            used_reference=use_ref,
            gate=self.backbone.last_gate,
            S=None if S is None else S.detach(),
            query_illum=self.last_query_illum_info,
            v3=self.backbone.last_v3_diag,
        )
        if c_t is not None:
            aux['c_t_stats'] = reliability_stats(c_t)
            aux['c_l_stats'] = reliability_stats(c_l)
            aux['match_diag'] = {k: (v.detach() if torch.is_tensor(v) else v)
                                 for k, v in (match_diag or {}).items()}
        self.last_aux = aux
        if return_aux:
            return out, aux
        return out

    def count_parameters(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_asrt(cfg: Optional[dict] = None) -> ASRT:
    cfg = dict(cfg or {})
    return ASRT(
        num_res_blocks=cfg.get('num_res_blocks', '8+8+4+2'),
        n_feats=int(cfg.get('n_feats', 64)),
        res_scale=float(cfg.get('res_scale', 1.0)),
        freeze_lte=bool(cfg.get('freeze_lte', True)),
        use_global_illum=bool(cfg.get('use_global_illum', False)),
        use_ref_illum=bool(cfg.get('use_ref_illum', False)),
        enable_lightup=bool(cfg.get('enable_lightup', True)),
        query_align=cfg.get('query_align'),
        enable_v3=bool(cfg.get('enable_v3', False)),
    )


def load_pretrained_into_v3(model: ASRT, state: dict) -> dict:
    """Load Stage A / B-v2 (non-v3) weights into v3.

    Every key in the checkpoint must exist in the v3 model with matching shape.
    Allowed missing keys in the *model* are new v3 modules only.
    """
    if not isinstance(state, dict):
        raise ValueError('state must be a state_dict')
    if 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    model_sd = model.state_dict()
    skipped_old = []
    shape_mismatch = []
    for k, v in state.items():
        if k not in model_sd:
            skipped_old.append(k)
        elif tuple(model_sd[k].shape) != tuple(v.shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
    if skipped_old:
        raise RuntimeError(
            'pretrained keys missing in v3 model (cannot skip old trunk): %s'
            % skipped_old[:20])
    if shape_mismatch:
        raise RuntimeError('shape mismatch loading pretrained: %s' % shape_mismatch[:10])

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError('unexpected keys from pretrained ckpt: %s' % list(unexpected)[:20])
    bad_missing = []
    for k in missing:
        if k.startswith('reliability.'):
            continue
        if k.startswith('backbone.coarse_illum_adapter.'):
            continue
        if k.startswith('backbone.middle_illum_adapter.'):
            continue
        if k.startswith('backbone.fine_ct_refine.'):
            continue
        if ('.adapter.' in k) and (
                k.startswith('backbone.RB11.') or k.startswith('backbone.RB22.')):
            continue
        bad_missing.append(k)
    if bad_missing:
        raise RuntimeError(
            'unexpected missing keys (not new v3 modules): %s' % bad_missing[:30])
    return dict(missing=list(missing), unexpected=list(unexpected), loaded=len(state))


def load_b_v2_into_v3(model: ASRT, state: dict) -> dict:
    """Alias for load_pretrained_into_v3 (backward compatible name)."""
    return load_pretrained_into_v3(model, state)
