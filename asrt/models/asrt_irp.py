"""ASRT Illumination–Reference Prior (vNext V1) top-level model."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from asrt.models.irp_csfi_unet import IRPCSFIUNet


class ASRTIRP(nn.Module):
    """End-to-end IRP restoration. Inputs/outputs RGB in [-1, 1]."""

    model_family = 'asrt_irp'
    model_version = 'irp_v1'
    is_irp = True
    is_v4 = False

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
        attention_type: str = 'window_spatial',
        **kwargs,
    ):
        super().__init__()
        self.prior_source = str(prior_source).lower()
        if self.prior_source not in ('none', 'low', 'nano'):
            raise ValueError('prior_source must be none|low|nano')
        self.n_feats = int(n_feats)
        self.num_res_blocks = str(num_res_blocks)
        self.enable_lightup = bool(enable_lightup)
        self.attn_heads = int(attn_heads)
        self.window_size = int(window_size)
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)
        self.ref_sigma = float(ref_sigma)
        self.attention_type = str(attention_type)
        if self.n_feats % self.attn_heads != 0:
            raise ValueError(
                'n_feats=%d not divisible by attention.heads=%d'
                % (self.n_feats, self.attn_heads))
        if self.window_size < 1:
            raise ValueError('window_size must be >= 1')
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
        self.irp_hparams = extract_irp_hparams_from_model(self)

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


def extract_irp_hparams_from_model(model: ASRTIRP) -> Dict[str, Any]:
    return dict(
        model_family=ASRTIRP.model_family,
        model_version=ASRTIRP.model_version,
        name='asrt_irp_v1',
        version='irp_v1',
        prior_source=str(model.prior_source),
        n_feats=int(model.n_feats),
        num_res_blocks=str(model.num_res_blocks),
        enable_lightup=bool(model.enable_lightup),
        attention_type=str(model.attention_type),
        attention_heads=int(model.attn_heads),
        attention_window_size=int(model.window_size),
        reference_prior_gaussian_sigma=float(model.ref_sigma),
        irp_alpha_init=float(model.alpha_init),
        irp_beta_init=float(model.beta_init),
    )


def extract_irp_hparams(cfg: Optional[dict] = None) -> Dict[str, Any]:
    cfg = dict(cfg or {})
    model_cfg = dict(cfg.get('model') or cfg)
    irp = dict(model_cfg.get('irp') or {})
    attn = dict(model_cfg.get('attention') or {})
    refp = dict(model_cfg.get('reference_prior') or {})
    prior_source = str(
        model_cfg.get('prior_source', irp.get('prior_source', 'nano'))).lower()
    return dict(
        model_family='asrt_irp',
        model_version='irp_v1',
        name=str(model_cfg.get('name', 'asrt_irp_v1')),
        version=str(model_cfg.get('version', 'irp_v1')),
        prior_source=prior_source,
        n_feats=int(model_cfg.get('n_feats', 64)),
        num_res_blocks=str(model_cfg.get('num_res_blocks', '8+8+4+2')),
        enable_lightup=bool(model_cfg.get('enable_lightup', True)),
        attention_type=str(attn.get('type', 'window_spatial')),
        attention_heads=int(attn.get('heads', 4)),
        attention_window_size=int(attn.get('window_size', 8)),
        reference_prior_gaussian_sigma=float(refp.get('gaussian_sigma', 7.0)),
        irp_alpha_init=float(irp.get('alpha_init', 1.0)),
        irp_beta_init=float(irp.get('beta_init', 0.1)),
    )


def build_asrt_irp(cfg: Optional[dict] = None) -> ASRTIRP:
    hp = extract_irp_hparams(cfg)
    return ASRTIRP(
        num_res_blocks=hp['num_res_blocks'],
        n_feats=hp['n_feats'],
        enable_lightup=hp['enable_lightup'],
        prior_source=hp['prior_source'],
        attn_heads=hp['attention_heads'],
        window_size=hp['attention_window_size'],
        alpha_init=hp['irp_alpha_init'],
        beta_init=hp['irp_beta_init'],
        ref_sigma=hp['reference_prior_gaussian_sigma'],
        attention_type=hp['attention_type'],
    )
