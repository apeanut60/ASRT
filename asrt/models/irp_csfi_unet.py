"""CSFI-UNet with Dual-Prior IRP Attention (ASRT vNext V1 backbone)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from asrt.models.attention import IlluminationEstimator
from asrt.models.csfi_unet import (
    CrossScaleCSFI2,
    CrossScaleCSFI3,
    MergeTail,
    ResBlock,
    SFE,
    parse_num_res_blocks,
    resize_like,
)
from asrt.models.illumination_prior_encoder import IlluminationPriorEncoder
from asrt.models.irp_attention import IRPTransformerBlock
from asrt.models.reference_prior_encoder import ReferencePriorEncoder


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)


def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1, bias=True)


class IRPCSFIUNet(nn.Module):
    """Cross-scale restore with Illumination/Reference Prior modulated attention.

    prior_source:
      - 'none': Illumination-only (A1); no ReferencePriorEncoder / no ref attn
      - 'low':  Dual-prior with Low as structural source (A2)
      - 'nano': Dual-prior with Nano Ref as structural source (A3)
    """

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
    ):
        super().__init__()
        blocks = parse_num_res_blocks(num_res_blocks)
        self.num_res_blocks = blocks
        self.n_feats = int(n_feats)
        self.enable_lightup = bool(enable_lightup)
        self.prior_source = str(prior_source).lower()
        if self.prior_source not in ('none', 'low', 'nano'):
            raise ValueError('prior_source must be none|low|nano')

        self.SFE = SFE(blocks[0], n_feats, res_scale)
        self.down1 = nn.Sequential(conv3x3(n_feats, n_feats, stride=2), nn.LeakyReLU(0.1))
        self.down2 = nn.Sequential(conv3x3(n_feats, n_feats, stride=2), nn.LeakyReLU(0.1))
        self.skip_e1 = nn.Sequential(conv1x1(n_feats * 2, n_feats), nn.LeakyReLU(0.1))
        self.skip_e0 = nn.Sequential(conv1x1(n_feats * 2, n_feats), nn.LeakyReLU(0.1))

        self.RB11 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[1])])
        self.conv11_tail = conv3x3(n_feats, n_feats)

        self.ex12 = CrossScaleCSFI2(n_feats)
        self.RB21 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[2])])
        self.RB22 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[2])])
        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        self.ex123 = CrossScaleCSFI3(n_feats)
        self.RB31 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.RB32 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.RB33 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        # Keep Low-only light-up path (legacy IlluminationEstimator), separate from IRP prior
        self.illumination_estimator = IlluminationEstimator(n_feats)
        self.illum_prior_encoder = IlluminationPriorEncoder(n_feats)

        dual = self.prior_source != 'none'
        self.ref_prior_encoder = (
            ReferencePriorEncoder(n_feats, sigma=ref_sigma) if dual else None)

        self.coarse_irp = IRPTransformerBlock(
            dim=n_feats, heads=attn_heads, window_size=window_size,
            use_reference=False, alpha_init=alpha_init, beta_init=beta_init)
        self.middle_irp = IRPTransformerBlock(
            dim=n_feats, heads=attn_heads, window_size=window_size,
            use_reference=dual, alpha_init=alpha_init, beta_init=beta_init)
        self.fine_irp = IRPTransformerBlock(
            dim=n_feats, heads=attn_heads, window_size=window_size,
            use_reference=dual, alpha_init=alpha_init, beta_init=beta_init)
        self.merge_tail = MergeTail(n_feats)

        self.last_used_reference = False
        self.last_aux = None

    @staticmethod
    def safe_pad(x, target_hw):
        ph = target_hw[0] - x.shape[-2]
        pw = target_hw[1] - x.shape[-1]
        if ph == 0 and pw == 0:
            return x
        mode = 'reflect' if ph < x.shape[-2] and pw < x.shape[-1] else 'replicate'
        return F.pad(x, (0, pw, 0, ph), mode=mode)

    def _run_rbs(self, feat, rbs, tail):
        start = feat
        for rb in rbs:
            feat = rb(feat)
        return start + tail(feat)

    def forward(
        self,
        low,
        ref=None,
        return_aux: bool = False,
        disable_reference: bool = False,
        collect_branch_stats: bool = False,
    ):
        """
        low/ref: RGB [-1,1]. Ref used only when prior_source=nano and ref is not None.
        When prior_source=low, external ref is ignored.

        disable_reference / collect_branch_stats are diagnostics only; default path
        math matches the pre-diagnostics forward.
        """
        want_block_aux = bool(return_aux or collect_branch_stats)
        orig_hw = low.shape[-2:]
        th = int(math.ceil(orig_hw[0] / 4.0) * 4)
        tw = int(math.ceil(orig_hw[1] / 4.0) * 4)
        x = self.safe_pad(low, (th, tw))
        if ref is not None:
            ref = self.safe_pad(
                F.interpolate(ref, size=orig_hw, mode='bilinear', align_corners=False)
                if tuple(ref.shape[-2:]) != orig_hw else ref,
                (th, tw))

        low01 = (x + 1.0) * 0.5
        illu_feat, log_gain = self.illumination_estimator(low01)
        if self.enable_lightup:
            lightup = low01 * torch.exp(log_gain.clamp(-1.5, 1.5))
            base = 2.0 * lightup - 1.0
        else:
            base = x

        # IRP illumination prior (independent of IlluminationEstimator)
        prior_i = self.illum_prior_encoder(x)

        # Reference / Low structural prior
        used_ref = False
        prior_r = None
        if self.prior_source == 'low':
            prior_r = self.ref_prior_encoder(x, x)
            used_ref = False  # not nano ref
        elif self.prior_source == 'nano' and ref is not None:
            prior_r = self.ref_prior_encoder(x, ref)
            used_ref = True
        # else: none or nano without ref → illumination-only at middle/fine
        self.last_used_reference = used_ref

        e0 = self.SFE(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)

        block_stats = {}

        def _run_irp(blk, feat, illum, rprior, tag):
            out = blk(
                feat, illumination_prior=illum, reference_prior=rprior,
                return_aux=want_block_aux,
                disable_reference=bool(disable_reference),
            )
            if want_block_aux:
                feat2, st = out
                block_stats[tag] = st
                return feat2
            return out

        # Coarse H/4 — illumination only (no TTSR inject heads)
        x11 = self._run_rbs(e2, self.RB11, self.conv11_tail)
        x11 = _run_irp(self.coarse_irp, x11, prior_i['h4'], None, 'h4')

        # Middle H/2 — CSFI topology matches CSFIUNet
        x21 = x11
        x22 = self.skip_e1(torch.cat([resize_like(x11, e1), e1], dim=1))
        x21, x22 = self.ex12(x21, x22)
        x21 = self._run_rbs(x21, self.RB21, self.conv21_tail)
        x22 = self._run_rbs(x22, self.RB22, self.conv22_tail)
        ref_h2 = prior_r['h2'] if prior_r is not None else None
        x22 = _run_irp(self.middle_irp, x22, prior_i['h2'], ref_h2, 'h2')

        # Fine H
        x31, x32 = x21, x22
        x33 = self.skip_e0(torch.cat([resize_like(x22, e0), e0], dim=1))
        x31, x32, x33 = self.ex123(x31, x32, x33)
        x31 = self._run_rbs(x31, self.RB31, self.conv31_tail)
        x32 = self._run_rbs(x32, self.RB32, self.conv32_tail)
        x33 = self._run_rbs(x33, self.RB33, self.conv33_tail)
        ref_h1 = prior_r['h1'] if prior_r is not None else None
        x33 = _run_irp(self.fine_irp, x33, prior_i['h1'], ref_h1, 'h1')

        residual = self.merge_tail(x31, x32, x33)
        out = (base + residual)[:, :, :orig_hw[0], :orig_hw[1]]
        aux = dict(
            used_reference=used_ref,
            prior_source=self.prior_source,
            prior_i_map=prior_i.get('map'),
            had_ref_prior=prior_r is not None,
            disable_reference=bool(disable_reference),
            branch_stats=block_stats if want_block_aux else None,
        )
        self.last_aux = aux
        if return_aux:
            return out, aux
        return out
