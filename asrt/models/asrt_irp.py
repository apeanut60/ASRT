"""ASRT Illumination–Reference Prior (vNext V1) top-level model."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from asrt.models.irp_csfi_unet import IRPCSFIUNet


class ASRTIRP(nn.Module):
    """End-to-end IRP restoration. Inputs/outputs RGB in [-1, 1]."""

    def __init__(
        self,
        num_res_blocks='8+8+4+2',
        n_feats=64,
        res_scale=1.0,
        enable_lightup=True,
        prior_source: str = 'nano',
        attn_heads: int = 4,
        window_size: int = 8,
        alpha_init: float = 1.0,
        beta_init: float = 0.1,
        ref_sigma: float = 7.0,
        **kwargs,
    ):
        super().__init__()
        self.prior_source = str(prior_source).lower()
        self.backbone = IRPCSFIUNet(
            num_res_blocks=num_res_blocks,
            n_feats=n_feats,
            res_scale=res_scale,
            enable_lightup=enable_lightup,
            prior_source=self.prior_source,
            attn_heads=attn_heads,
            window_size=window_size,
            alpha_init=alpha_init,
            beta_init=beta_init,
            ref_sigma=ref_sigma,
        )
        self.last_aux = None

    def forward(self, low, ref=None, return_aux: bool = False):
        out = self.backbone(low, ref=ref, return_aux=return_aux)
        if return_aux:
            pred, aux = out
            self.last_aux = aux
            return pred, aux
        self.last_aux = self.backbone.last_aux
        return out

    def count_parameters(self) -> dict:
        def n(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else 0

        bb = self.backbone
        return dict(
            total=n(self),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            illum_prior_encoder=n(bb.illum_prior_encoder),
            ref_prior_encoder=n(bb.ref_prior_encoder),
            coarse_irp=n(bb.coarse_irp),
            middle_irp=n(bb.middle_irp),
            fine_irp=n(bb.fine_irp),
            illumination_estimator=n(bb.illumination_estimator),
        )


def build_asrt_irp(cfg: Optional[dict] = None) -> ASRTIRP:
    cfg = dict(cfg or {})
    model_cfg = dict(cfg.get('model') or cfg)
    irp = dict(model_cfg.get('irp') or {})
    attn = dict(model_cfg.get('attention') or {})
    refp = dict(model_cfg.get('reference_prior') or {})
    prior_source = str(
        model_cfg.get('prior_source', irp.get('prior_source', 'nano'))).lower()
    return ASRTIRP(
        num_res_blocks=model_cfg.get('num_res_blocks', '8+8+4+2'),
        n_feats=int(model_cfg.get('n_feats', 64)),
        res_scale=float(model_cfg.get('res_scale', 1.0)),
        enable_lightup=bool(model_cfg.get('enable_lightup', True)),
        prior_source=prior_source,
        attn_heads=int(attn.get('heads', 4)),
        window_size=int(attn.get('window_size', 8)),
        alpha_init=float(irp.get('alpha_init', 1.0)),
        beta_init=float(irp.get('beta_init', 0.1)),
        ref_sigma=float(refp.get('gaussian_sigma', 7.0)),
    )
