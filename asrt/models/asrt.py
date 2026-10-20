"""Top-level ASRT model: dual-input with true no-reference bypass."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from asrt.models.csfi_unet import CSFIUNet, parse_num_res_blocks
from asrt.models.reference import LTE, SearchTransfer


class ASRT(nn.Module):
    """Illumination-Guided Cross-Scale Reference Restoration.

    forward(low, ref=None, return_aux=False) -> enhanced [-1,1]
    When ref is None, LTE/SearchTransfer/ref inject/RefIllum are skipped.
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
    ):
        super().__init__()
        self.backbone = CSFIUNet(
            num_res_blocks=num_res_blocks,
            n_feats=n_feats,
            res_scale=res_scale,
            igab_blocks=igab_blocks,
            igab_heads=igab_heads,
            enable_lightup=enable_lightup,
            use_global_illum=use_global_illum,
            use_ref_illum=use_ref_illum,
        )
        self.lte = LTE(requires_grad=not freeze_lte)
        self.search_transfer = SearchTransfer()
        self.freeze_lte = bool(freeze_lte)
        self.last_aux = None

    @staticmethod
    def _resize_to(ref, hw):
        if tuple(ref.shape[-2:]) == tuple(hw):
            return ref
        return F.interpolate(ref, size=tuple(hw), mode='bilinear', align_corners=False)

    def match_reference(self, low, ref):
        """Run LTE + SearchTransfer. Inputs in [-1, 1], same spatial size preferred."""
        ref = self._resize_to(ref, low.shape[-2:])
        # Pad to 4x so LTE/SearchTransfer grids align with backbone pad.
        import math
        th = int(math.ceil(low.shape[-2] / 4.0) * 4)
        tw = int(math.ceil(low.shape[-1] / 4.0) * 4)
        low_p = CSFIUNet.safe_pad(low, (th, tw))
        ref_p = CSFIUNet.safe_pad(ref, (th, tw))
        q01 = (low_p + 1.) * 0.5
        r01 = (ref_p + 1.) * 0.5
        _, _, q_lv3 = self.lte(q01)
        ref_lv1, ref_lv2, ref_lv3 = self.lte(r01)
        _, _, k_lv3 = self.lte(r01)
        S, T3, T2, T1 = self.search_transfer(q_lv3, k_lv3, ref_lv1, ref_lv2, ref_lv3)
        return S, T3, T2, T1, ref_p

    def forward(self, low, ref=None, return_aux=False):
        if low.dim() != 4 or low.size(1) != 3:
            raise ValueError('low must be [B,3,H,W]')
        use_ref = ref is not None
        S = T3 = T2 = T1 = None
        ref_aligned = None
        if use_ref:
            if ref.dim() != 4 or ref.size(1) != 3:
                raise ValueError('ref must be [B,3,H,W] or None')
            if ref.size(0) != low.size(0):
                raise ValueError('ref/low batch mismatch')
            S, T3, T2, T1, ref_aligned = self.match_reference(low, ref)

        out = self.backbone(
            low, S=S, T_lv3=T3, T_lv2=T2, T_lv1=T1,
            ref=ref_aligned if use_ref else None,
            use_reference=use_ref,
        )
        aux = dict(
            used_reference=use_ref,
            gate=self.backbone.last_gate,
            S=None if S is None else S.detach(),
        )
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
    )
