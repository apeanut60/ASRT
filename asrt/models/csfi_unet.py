"""CSFI-U-Net backbone: cross-scale restore with optional reference injection."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from asrt.models.attention import IGAB, IlluminationEstimator
from asrt.models.cond_resblock import CondResBlock
from asrt.models.illumination_prior import IlluminationAdapter
from asrt.models.legacy_heads import GlobalIllumHead, RefIllumTransfer
from asrt.models.reliability import FineCTRefine
from asrt.models.v4_adapters import IlluminationAdapter as IllumAdapterV4
from asrt.models.v4_adapters import TextureAdapter


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)


def conv3x3(in_ch, out_ch, stride=1, dilation=1):
    return nn.Conv2d(
        in_ch, out_ch, 3, stride=stride, padding=dilation, dilation=dilation, bias=True)


def resize_like(x, target, mode='bilinear'):
    hw = target.shape[-2:]
    if x.shape[-2:] == hw:
        return x
    return F.interpolate(x, size=hw, mode=mode, align_corners=False)


class ResBlock(nn.Module):
    def __init__(self, channels, res_scale=1.):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)

    def forward(self, x):
        return x + self.res_scale * self.conv2(self.relu(self.conv1(x)))


class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super().__init__()
        self.conv_head = conv3x3(3, n_feats)
        self.RBs = nn.ModuleList([
            ResBlock(n_feats, res_scale=res_scale) for _ in range(num_res_blocks)])
        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x0 = x
        for rb in self.RBs:
            x = rb(x)
        return self.conv_tail(x) + x0


class CrossScaleCSFI2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, stride=2)
        self.conv_merge1 = conv3x3(2 * n_feats, n_feats)
        self.conv_merge2 = conv3x3(2 * n_feats, n_feats)

    def forward(self, x1, x2):
        x12 = F.relu(self.conv12(resize_like(x1, x2)))
        x21 = resize_like(F.relu(self.conv21(x2)), x1)
        y1 = F.relu(self.conv_merge1(torch.cat([x1, x21], dim=1)))
        y2 = F.relu(self.conv_merge2(torch.cat([x2, x12], dim=1)))
        return y1, y2


class CrossScaleCSFI3(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        c = n_feats
        self.conv12 = conv1x1(c, c)
        self.conv13 = conv1x1(c, c)
        self.conv21 = conv3x3(c, c, stride=2)
        self.conv23 = conv1x1(c, c)
        self.conv31_1 = conv3x3(c, c, stride=2)
        self.conv31_2 = conv3x3(c, c, stride=2)
        self.conv32 = conv3x3(c, c, stride=2)
        self.conv_merge1 = conv3x3(3 * c, c)
        self.conv_merge2 = conv3x3(3 * c, c)
        self.conv_merge3 = conv3x3(3 * c, c)

    def forward(self, x1, x2, x3):
        x12 = F.relu(self.conv12(resize_like(x1, x2)))
        x13 = F.relu(self.conv13(resize_like(x1, x3)))
        x21 = resize_like(F.relu(self.conv21(x2)), x1)
        x23 = F.relu(self.conv23(resize_like(x2, x3)))
        x31 = resize_like(F.relu(self.conv31_2(F.relu(self.conv31_1(x3)))), x1)
        x32 = resize_like(F.relu(self.conv32(x3)), x2)
        y1 = F.relu(self.conv_merge1(torch.cat([x1, x21, x31], dim=1)))
        y2 = F.relu(self.conv_merge2(torch.cat([x2, x12, x32], dim=1)))
        y3 = F.relu(self.conv_merge3(torch.cat([x3, x13, x23], dim=1)))
        return y1, y2, y3


class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv_merge = conv3x3(3 * n_feats, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, 3)
        with torch.no_grad():
            self.conv_tail2.weight.mul_(0.01)
            self.conv_tail2.bias.zero_()

    def forward(self, x1, x2, x3):
        f = F.relu(self.conv_merge(torch.cat([
            resize_like(x1, x3), resize_like(x2, x3), x3], dim=1)))
        return self.conv_tail2(self.conv_tail1(f))


def parse_num_res_blocks(spec) -> list:
    if isinstance(spec, (list, tuple)):
        blocks = [int(x) for x in spec]
    else:
        blocks = [int(x) for x in str(spec).split('+')]
    if len(blocks) != 4:
        raise ValueError('num_res_blocks expected SFE+coarse+middle+fine, got %r' % (spec,))
    return blocks


class CSFIUNet(nn.Module):
    """Cross-scale CSFI U-Net with IGAB. Accepts optional S/T from reference."""

    def __init__(
        self,
        num_res_blocks='8+8+4+2',
        n_feats=64,
        res_scale=1.,
        igab_blocks=(1, 1),
        igab_heads=1,
        enable_lightup=True,
        use_global_illum=False,
        use_ref_illum=False,
        ref_illum_pool=8,
        enable_v3: bool = False,
        enable_v4: bool = False,
    ):
        super().__init__()
        blocks = parse_num_res_blocks(num_res_blocks)
        if n_feats % igab_heads:
            raise ValueError('n_feats must be divisible by igab_heads')
        if enable_v3 and enable_v4:
            raise ValueError('enable_v3 and enable_v4 are mutually exclusive')
        self.num_res_blocks = blocks
        self.n_feats = n_feats
        self.enable_lightup = enable_lightup
        self.use_global_illum = bool(use_global_illum)
        self.use_ref_illum = bool(use_ref_illum)
        self.enable_v3 = bool(enable_v3)
        self.enable_v4 = bool(enable_v4)

        self.SFE = SFE(blocks[0], n_feats, res_scale)
        self.down1 = nn.Sequential(conv3x3(n_feats, n_feats, stride=2), nn.LeakyReLU(0.1))
        self.down2 = nn.Sequential(conv3x3(n_feats, n_feats, stride=2), nn.LeakyReLU(0.1))
        self.skip_e1 = nn.Sequential(conv1x1(n_feats * 2, n_feats), nn.LeakyReLU(0.1))
        self.skip_e0 = nn.Sequential(conv1x1(n_feats * 2, n_feats), nn.LeakyReLU(0.1))

        self.conv11_head = conv3x3(n_feats + 256, n_feats)
        self.RB11 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[1])])
        self.conv11_tail = conv3x3(n_feats, n_feats)

        self.conv22_head = conv3x3(n_feats + 128, n_feats)
        self.ex12 = CrossScaleCSFI2(n_feats)
        self.RB21 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[2])])
        self.RB22 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[2])])
        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        self.conv33_head = conv3x3(n_feats + 64, n_feats)
        self.ex123 = CrossScaleCSFI3(n_feats)
        self.RB31 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.RB32 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.RB33 = nn.ModuleList([ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        # Soft init for reference injectors: near-identity when gate*head≈0.
        for head in (self.conv11_head, self.conv22_head, self.conv33_head):
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.weight.mul_(0.01)

        self.illumination_estimator = IlluminationEstimator(n_feats)
        self.coarse_igab = IGAB(n_feats, num_blocks=int(igab_blocks[0]), heads=igab_heads)
        self.middle_igab = IGAB(n_feats, num_blocks=int(igab_blocks[1]), heads=igab_heads)
        self.merge_tail = MergeTail(n_feats)
        self.global_illum_head = GlobalIllumHead()
        self.ref_illum = RefIllumTransfer(pool=ref_illum_pool)

        # ASRT-v3 modules (optional). Last Coarse/Middle RB → CondResBlock.
        self.coarse_illum_adapter = None
        self.middle_illum_adapter = None
        self.fine_ct_refine = None
        if self.enable_v3:
            self.RB11[-1] = CondResBlock(n_feats, res_scale)
            self.RB22[-1] = CondResBlock(n_feats, res_scale)
            self.coarse_illum_adapter = IlluminationAdapter(n_feats)
            self.middle_illum_adapter = IlluminationAdapter(n_feats)
            self.fine_ct_refine = FineCTRefine()

        # ASRT-v4: TextureAdapters + IllumAdapters; keep plain ResBlocks / IGAB.
        self.tex_adapt_h4 = None
        self.tex_adapt_h2 = None
        self.tex_adapt_h1 = None
        self.illum_adapt_h4 = None
        self.illum_adapt_h2 = None
        if self.enable_v4:
            self.tex_adapt_h4 = TextureAdapter(n_feats)
            self.tex_adapt_h2 = TextureAdapter(n_feats)
            self.tex_adapt_h1 = TextureAdapter(n_feats)
            self.illum_adapt_h4 = IllumAdapterV4(n_feats)
            self.illum_adapt_h2 = IllumAdapterV4(n_feats)

        self.last_gate = None
        self.last_used_reference = False
        self.last_v3_diag = None
        self.last_v4_diag = None

    @staticmethod
    def safe_pad(x, target_hw):
        ph = target_hw[0] - x.shape[-2]
        pw = target_hw[1] - x.shape[-1]
        if ph == 0 and pw == 0:
            return x
        mode = 'reflect' if ph < x.shape[-2] and pw < x.shape[-1] else 'replicate'
        return F.pad(x, (0, pw, 0, ph), mode=mode)

    @staticmethod
    def _align_ref(T, feat, name):
        if T is None:
            raise ValueError(name + ' required when using reference')
        dh = abs(T.shape[-2] - feat.shape[-2])
        dw = abs(T.shape[-1] - feat.shape[-1])
        if dh > 3 or dw > 3:
            raise ValueError(
                '%s spatial mismatch >3px: %s vs %s' % (name, T.shape[-2:], feat.shape[-2:]))
        return resize_like(T, feat)

    @staticmethod
    def _gate(S, feat):
        if S is None:
            return None
        return torch.sigmoid(F.interpolate(
            S, size=feat.shape[-2:], mode='bicubic', align_corners=False))

    def _inject(self, feat, T, S, head, use_ref: bool, c_t=None):
        if not use_ref or T is None or S is None:
            return feat, None
        T = self._align_ref(T, feat, 'T')
        g = self._gate(S, feat)
        residual = g * head(torch.cat([feat, T], dim=1))
        if c_t is not None:
            ct = resize_like(c_t, feat)
            residual = ct * residual
        return feat + residual, residual

    def _tex_correct(self, feat, T, c_t, adapter):
        """Reliability-aware texture correction: F + C_T * delta(F,T)."""
        if adapter is None or T is None or c_t is None:
            return feat, None
        T = self._align_ref(T, feat, 'T_v4')
        ct = resize_like(c_t, feat)
        before = feat
        out = adapter(feat, T, ct)
        return out, out - before

    def _run_rbs(self, feat, rbs, tail, l_guided=None, c_l=None):
        start = feat
        for rb in rbs:
            if isinstance(rb, CondResBlock):
                feat = rb(feat, l_guided, c_l)
            else:
                feat = rb(feat)
        return start + tail(feat)

    def encode_low_stem(self, x):
        """Pad + illum + SFE + downs. Used by ASRTv4 for Query features."""
        orig_hw = x.shape[-2:]
        th = int(math.ceil(orig_hw[0] / 4.0) * 4)
        tw = int(math.ceil(orig_hw[1] / 4.0) * 4)
        low = self.safe_pad(x, (th, tw))
        low01 = (low + 1.) * 0.5
        illu_h, log_gain = self.illumination_estimator(low01)
        if self.enable_lightup:
            lightup = low01 * torch.exp(log_gain.clamp(-1.5, 1.5))
            base = 2. * lightup - 1.
        else:
            base = low
        e0 = self.SFE(low)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        illu_h2 = resize_like(illu_h, e1)
        illu_h4 = resize_like(illu_h, e2)
        return dict(
            low=low, low01=low01, base=base, orig_hw=orig_hw,
            e0=e0, e1=e1, e2=e2,
            illu_h=illu_h, illu_h2=illu_h2, illu_h4=illu_h4,
            log_gain=log_gain,
        )

    def forward(
        self,
        x,
        S=None,
        T_lv3=None,
        T_lv2=None,
        T_lv1=None,
        ref=None,
        use_reference: Optional[bool] = None,
        c_t=None,
        c_l=None,
        illum_prior=None,
        stem: Optional[dict] = None,
    ):
        """
        Args:
            x: Low RGB in [-1, 1]
            S/T_*: SearchTransfer outputs (legacy/v3); ignored when use_reference is False
            use_reference: if None, inferred from available tensors
            c_t / c_l: reliability maps [B,1,H/4,W/4]
            illum_prior: dict with p_ref_matched / p_base / d_ref
            stem: optional precomputed encode_low_stem (v4 shares Query stem)
        """
        if use_reference is None:
            use_reference = (
                (T_lv3 is not None and T_lv2 is not None and T_lv1 is not None)
                and (S is not None or self.enable_v4))
        self.last_used_reference = bool(use_reference)
        use_v3 = bool(
            self.enable_v3 and use_reference and c_t is not None and c_l is not None
            and illum_prior is not None)
        use_v4 = bool(
            self.enable_v4 and use_reference and c_t is not None and c_l is not None
            and illum_prior is not None
            and T_lv3 is not None and T_lv2 is not None and T_lv1 is not None)

        if stem is None:
            stem = self.encode_low_stem(x)
        low = stem['low']
        base = stem['base']
        orig_hw = stem['orig_hw']
        e0, e1, e2 = stem['e0'], stem['e1'], stem['e2']
        illu_h2, illu_h4 = stem['illu_h2'], stem['illu_h4']

        # Guided illumination
        l_guided_h4 = illu_h4
        l_guided_h2 = illu_h2
        delta_c = delta_m = None
        if use_v3:
            l_guided_h4, delta_c = self.coarse_illum_adapter(
                illu_h4,
                illum_prior['p_ref_matched'],
                illum_prior['p_base'],
                illum_prior['d_ref'],
                c_l,
            )
            l_guided_h2, delta_m = self.middle_illum_adapter(
                illu_h2,
                illum_prior['p_ref_matched'],
                illum_prior['p_base'],
                illum_prior['d_ref'],
                c_l,
            )
        elif use_v4:
            l_guided_h4 = self.illum_adapt_h4(
                illu_h4, illum_prior['p_ref_matched'],
                illum_prior['d_ref'], c_l)
            l_guided_h2 = self.illum_adapt_h2(
                illu_h2, illum_prior['p_ref_matched'],
                illum_prior['d_ref'], c_l)
            delta_c = l_guided_h4 - illu_h4
            delta_m = l_guided_h2 - illu_h2

        c_t_h4 = c_t if (use_v3 or use_v4) else None
        c_t_h2 = resize_like(c_t, e1) if (use_v3 or use_v4) else None
        c_t_h1 = None
        if use_v3:
            c_t_up = resize_like(c_t, e0)
            c_t_h1 = self.fine_ct_refine(c_t_up, low)
        elif use_v4:
            c_t_h1 = resize_like(c_t, e0)

        res11 = res22 = res33 = None
        if use_v4:
            x11, res11 = self._tex_correct(e2, T_lv3, c_t_h4, self.tex_adapt_h4)
        else:
            x11, res11 = self._inject(e2, T_lv3, S, self.conv11_head, use_reference, c_t_h4)
        x11 = self._run_rbs(
            x11, self.RB11, self.conv11_tail,
            l_guided=l_guided_h4 if use_v3 else None,
            c_l=c_l if use_v3 else None)
        x11 = self.coarse_igab(x11, l_guided_h4)

        x21 = x11
        x22 = self.skip_e1(torch.cat([resize_like(x11, e1), e1], dim=1))
        if use_v4:
            x22, res22 = self._tex_correct(x22, T_lv2, c_t_h2, self.tex_adapt_h2)
        else:
            x22, res22 = self._inject(x22, T_lv2, S, self.conv22_head, use_reference, c_t_h2)
        x21, x22 = self.ex12(x21, x22)
        x21 = self._run_rbs(x21, self.RB21, self.conv21_tail)
        x22 = self._run_rbs(
            x22, self.RB22, self.conv22_tail,
            l_guided=l_guided_h2 if use_v3 else None,
            c_l=c_l if use_v3 else None)
        x22 = self.middle_igab(x22, l_guided_h2)

        x31, x32 = x21, x22
        x33 = self.skip_e0(torch.cat([resize_like(x22, e0), e0], dim=1))
        if use_v4:
            x33, res33 = self._tex_correct(x33, T_lv1, c_t_h1, self.tex_adapt_h1)
        else:
            x33, res33 = self._inject(x33, T_lv1, S, self.conv33_head, use_reference, c_t_h1)
        x31, x32, x33 = self.ex123(x31, x32, x33)
        x31 = self._run_rbs(x31, self.RB31, self.conv31_tail)
        x32 = self._run_rbs(x32, self.RB32, self.conv32_tail)
        x33 = self._run_rbs(x33, self.RB33, self.conv33_tail)

        rgb_residual = self.merge_tail(x31, x32, x33)
        out = (base + rgb_residual)[..., :orig_hw[0], :orig_hw[1]]

        if self.use_ref_illum and ref is not None:
            out = self.ref_illum(out, x, ref)
        if self.use_global_illum:
            out = self.global_illum_head.apply(out, x)

        g = self._gate(S, e0)
        self.last_gate = None if g is None else g.detach()
        if use_v3:
            self.last_v3_diag = dict(
                c_t=c_t.detach(),
                c_l=c_l.detach(),
                c_t_fine=None if c_t_h1 is None else c_t_h1.detach(),
                delta_l_coarse=None if delta_c is None else delta_c.detach(),
                delta_l_middle=None if delta_m is None else delta_m.detach(),
                inject_res_norm=dict(
                    coarse=float(res11.detach().norm()) if res11 is not None else 0.0,
                    middle=float(res22.detach().norm()) if res22 is not None else 0.0,
                    fine=float(res33.detach().norm()) if res33 is not None else 0.0,
                ),
                base=base.detach(),
            )
        else:
            self.last_v3_diag = None
        if use_v4:
            self.last_v4_diag = dict(
                c_t=c_t.detach(),
                c_l=c_l.detach(),
                delta_l_coarse=None if delta_c is None else delta_c.detach(),
                delta_l_middle=None if delta_m is None else delta_m.detach(),
                tex_res_norm=dict(
                    coarse=float(res11.detach().norm()) if res11 is not None else 0.0,
                    middle=float(res22.detach().norm()) if res22 is not None else 0.0,
                    fine=float(res33.detach().norm()) if res33 is not None else 0.0,
                ),
                l_guided_diff=float((l_guided_h4 - illu_h4).detach().norm()),
                base=base.detach(),
            )
        else:
            self.last_v4_diag = None
        return out
