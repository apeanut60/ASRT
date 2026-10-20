"""TTSR-lowlight v3: reference-guided cross-scale CSFI U-Net.

Designed as a drop-in MainNet for model/TTSREnhance.py.  Low and output use
[-1, 1], matching the existing TTSR-lowlight pipeline.  Unlike the old
same-resolution MainNet, the three streams live at H/4, H/2, and H.

Reference tensors from the existing SearchTransfer are used at *native*
levels: T_lv3 ~ [B,256,H/4,W/4], T_lv2 ~ [B,128,H/2,W/2],
T_lv1 ~ [B,64,H,W].  This module does not generate or alter references.

Architecture:
  Low -> SFE -> E0 (H) -> E1 (H/2) -> E2 (H/4)
  coarse : E2 + T_lv3 -> RB -> IGAB(illum_h4)
  middle : Up(coarse)+E1 + T_lv2 -> CSFI2 -> RB -> IGAB(illum_h2)
  fine   : Up(middle)+E0 + T_lv1 -> CSFI3 -> RB
  Merge(coarse,middle,fine) -> RGB residual + illuminated low

IG-MSA matches the official Retinexformer channel-attention semantics:
  attention matrix is [head, dim_head, dim_head], NOT [HW, HW].

References:
  researchmm/TTSR model/MainNet.py  (original cross-scale CSFI)
  caiyuanhao1998/Retinexformer basicsr/models/archs/RetinexFormer_arch.py
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_ch: int, out_ch: int, stride: int = 1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                     padding=dilation, dilation=dilation, bias=True)


class ResBlock(nn.Module):
    """Keep legacy names and parameter shapes for partial weight transfer."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 res_scale=1.):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            raise ValueError('Legacy ResBlock requires matching channels, stride=1')
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        return x + self.res_scale * self.conv2(self.relu(self.conv1(x)))


class SFE(nn.Module):
    """Legacy SFE, state_dict-compatible with MainNetEnhance.SFE."""

    def __init__(self, num_res_blocks, n_feats, res_scale):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)
        self.RBs = nn.ModuleList([
            ResBlock(n_feats, n_feats, res_scale=res_scale)
            for _ in range(num_res_blocks)
        ])
        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x0 = x
        for rb in self.RBs:
            x = rb(x)
        return self.conv_tail(x) + x0


def resize_like(x: torch.Tensor, target: torch.Tensor,
                mode: str = 'bilinear') -> torch.Tensor:
    hw = target.shape[-2:]
    if x.shape[-2:] == hw:
        return x
    return F.interpolate(x, size=hw, mode=mode, align_corners=False)


class CrossScaleCSFI2(nn.Module):
    """TTSR-style bidirectional exchange: coarse H/4 <-> mid H/2."""

    def __init__(self, n_feats):
        super().__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, stride=2)
        self.conv_merge1 = conv3x3(2 * n_feats, n_feats)
        self.conv_merge2 = conv3x3(2 * n_feats, n_feats)

    def forward(self, x1, x2):
        x12 = F.relu(self.conv12(resize_like(x1, x2)))
        x21 = F.relu(self.conv21(x2))
        x21 = resize_like(x21, x1)
        y1 = F.relu(self.conv_merge1(torch.cat([x1, x21], dim=1)))
        y2 = F.relu(self.conv_merge2(torch.cat([x2, x12], dim=1)))
        return y1, y2


class CrossScaleCSFI3(nn.Module):
    """All-to-all scale exchange: H/4, H/2, H, adapted from original TTSR."""

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


class IlluminationEstimator(nn.Module):
    """Retinexformer-inspired lightweight estimator, evaluated on Low [0,1].

    Outputs illumination embedding for IGAB and log-gain for light-up.
    The log-gain head starts at 0 (identity light-up), retaining a usable
    gradient from step 1.  The illumination embedding does not depend on the
    log-gain head and is trained from the attention path immediately.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, channels, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(channels, channels, kernel_size=5,
                                    padding=2, groups=channels, bias=True)
        self.conv2 = nn.Conv2d(channels, 3, kernel_size=1, bias=True)
        # Identity initialization for light-up; IGAB path still has gradients.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, low01):
        mean_c = low01.mean(dim=1, keepdim=True)
        f = self.depth_conv(self.conv1(torch.cat([low01, mean_c], dim=1)))
        log_gain = self.conv2(f)
        return f, log_gain


class IG_MSA(nn.Module):
    """Native channel-transposed illumination-guided attention (Retinexformer)."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 1):
        super().__init__()
        if dim != dim_head * heads:
            raise ValueError('IG_MSA needs dim == heads*dim_head; got %d vs %d' %
                             (dim, heads * dim_head))
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x, illu):
        b, h, w, c = x.shape
        if illu.shape != x.shape:
            raise ValueError('IG_MSA: x and illu must have identical BHWC shapes')
        n = h * w
        q = self.to_q(x.reshape(b, n, c))
        k = self.to_k(x.reshape(b, n, c))
        v_inp = self.to_v(x.reshape(b, n, c))
        # [B, head, n, d] -> [B, head, d, n]
        def reshape(t):
            return t.reshape(b, n, self.num_heads, self.dim_head).permute(0, 2, 3, 1)
        q, k, v = reshape(q), reshape(k), reshape(v_inp)
        illum = reshape(illu.reshape(b, n, c))
        v = v * illum
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = torch.matmul(k, q.transpose(-2, -1))
        attn = torch.softmax(attn * self.rescale, dim=-1)
        out = torch.matmul(attn, v)  # [B, head, d, n]
        out = out.permute(0, 3, 1, 2).reshape(b, n, c)
        out_c = self.proj(out).reshape(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2))
        return out_c + out_p.permute(0, 2, 3, 1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, padding=1,
                      groups=dim * mult, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class IGAB(nn.Module):
    """Official-style IGAB: (IG-MSA + residual) then (LayerNorm-FFN + residual)."""

    def __init__(self, dim: int, num_blocks: int = 1, heads: int = 1):
        super().__init__()
        if dim % heads:
            raise ValueError('channels must be divisible by heads')
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim // heads, heads=heads),
                nn.LayerNorm(dim),
                FeedForward(dim),
            ]) for _ in range(num_blocks)
        ])

    def forward(self, feat, illu):
        x = feat.permute(0, 2, 3, 1)
        illu = illu.permute(0, 2, 3, 1)
        for attn, norm, ff in self.blocks:
            x = x + attn(x, illu)
            x = x + ff(norm(x))
        return x.permute(0, 3, 1, 2)


class MergeTail(nn.Module):
    """Merge native pyramid to full-res features and predict a RGB residual."""

    def __init__(self, n_feats):
        super().__init__()
        self.conv_merge = conv3x3(3 * n_feats, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, 3)
        # Output is a residual on top of illuminated Low; near-zero init
        # prevents a large random RGB offset while preserving backprop.
        with torch.no_grad():
            self.conv_tail2.weight.mul_(0.01)
            self.conv_tail2.bias.zero_()

    def forward(self, x1, x2, x3):
        u1 = resize_like(x1, x3)
        u2 = resize_like(x2, x3)
        f = F.relu(self.conv_merge(torch.cat([u1, u2, x3], dim=1)))
        return self.conv_tail2(self.conv_tail1(f))


class GlobalIllumHead(nn.Module):
    """Copied interface / weight shapes from legacy MainNetEnhance."""

    def __init__(self, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc_gain = nn.Linear(hidden, 3)
        self.fc_bias = nn.Linear(hidden, 3)
        self.fc_gamma = nn.Linear(hidden, 3)
        for fc in (self.fc_gain, self.fc_bias, self.fc_gamma):
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, low):
        m = low.mean(dim=(2, 3))
        s = low.std(dim=(2, 3), unbiased=False)
        h = self.act(self.fc1(torch.cat([m, s], dim=1)))
        lg = self.fc_gain(h).clamp(-1.5, 1.5)
        gain = torch.exp(lg).view(-1, 3, 1, 1)
        bias = self.fc_bias(h).clamp(-0.5, 0.5).view(-1, 3, 1, 1)
        gamma = torch.exp(self.fc_gamma(h).clamp(-0.7, 0.7)).view(-1, 3, 1, 1)
        return gain, bias, gamma


class RefIllumTransfer(nn.Module):
    """Legacy low-frequency reference tone head, kept for API/checkpoint compatibility."""

    def __init__(self, n_feats=16, pool=8, init_scale=0.05):
        super().__init__()
        self.pool = max(1, int(pool))
        self.conv1 = conv3x3(9, n_feats)
        self.conv2 = conv3x3(n_feats, n_feats)
        self.conv3 = conv3x3(n_feats, 3)
        with torch.no_grad():
            self.conv3.weight.mul_(init_scale)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x, low, ref):
        if ref is None:
            return x
        p = self.pool
        # Adaptive pooling guarantees shape consistency for small/odd images.
        pooled_hw = (max(1, x.shape[-2] // p), max(1, x.shape[-1] // p))
        xr = F.adaptive_avg_pool2d(x, pooled_hw)
        lr = F.adaptive_avg_pool2d(low, pooled_hw)
        rr = F.adaptive_avg_pool2d(ref, pooled_hw)
        h = F.relu(self.conv1(torch.cat([rr, lr, xr], dim=1)))
        h = F.relu(self.conv2(h))
        delta = resize_like(self.conv3(h), x)
        return torch.clamp(x + delta, -1, 1)


class MainNetCSFIUNet(nn.Module):
    """Drop-in MainNet with a *real* three-scale U-Net + CSFI + IGAB.

    Arguments match legacy MainNetEnhance(...); forward signature retains
    all legacy branch-bypass/diagnostic kwargs used by TTSREnhance and trainer.
    """

    def __init__(self, num_res_blocks, n_feats=64, res_scale=1.,
                 ref_illum_pool=8, igab_blocks=(1, 1), igab_heads=1,
                 enable_lightup=True):
        super().__init__()
        if len(num_res_blocks) != 4:
            raise ValueError('num_res_blocks expected [SFE, coarse, middle, fine]')
        if n_feats % igab_heads != 0:
            raise ValueError('n_feats must be divisible by igab_heads')
        self.num_res_blocks = list(num_res_blocks)
        self.n_feats = n_feats
        self.enable_lightup = enable_lightup

        # Encoder / skip pyramid.
        self.SFE = SFE(num_res_blocks[0], n_feats, res_scale)
        self.down1 = nn.Sequential(conv3x3(n_feats, n_feats, stride=2), nn.LeakyReLU(.1))
        self.down2 = nn.Sequential(conv3x3(n_feats, n_feats, stride=2), nn.LeakyReLU(.1))
        self.skip_e1 = nn.Sequential(conv1x1(n_feats * 2, n_feats), nn.LeakyReLU(.1))
        self.skip_e0 = nn.Sequential(conv1x1(n_feats * 2, n_feats), nn.LeakyReLU(.1))

        # Legacy stage names for weight transfer where compatible.
        self.conv11_head = conv3x3(n_feats + 256, n_feats)
        self.RB11 = nn.ModuleList([ResBlock(n_feats, n_feats, res_scale=res_scale)
                                   for _ in range(num_res_blocks[1])])
        self.conv11_tail = conv3x3(n_feats, n_feats)

        self.conv22_head = conv3x3(n_feats + 128, n_feats)
        self.ex12 = CrossScaleCSFI2(n_feats)
        self.RB21 = nn.ModuleList([ResBlock(n_feats, n_feats, res_scale=res_scale)
                                   for _ in range(num_res_blocks[2])])
        self.RB22 = nn.ModuleList([ResBlock(n_feats, n_feats, res_scale=res_scale)
                                   for _ in range(num_res_blocks[2])])
        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        self.conv33_head = conv3x3(n_feats + 64, n_feats)
        self.ex123 = CrossScaleCSFI3(n_feats)
        self.RB31 = nn.ModuleList([ResBlock(n_feats, n_feats, res_scale=res_scale)
                                   for _ in range(num_res_blocks[3])])
        self.RB32 = nn.ModuleList([ResBlock(n_feats, n_feats, res_scale=res_scale)
                                   for _ in range(num_res_blocks[3])])
        self.RB33 = nn.ModuleList([ResBlock(n_feats, n_feats, res_scale=res_scale)
                                   for _ in range(num_res_blocks[3])])
        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        self.illumination_estimator = IlluminationEstimator(n_feats)
        self.coarse_igab = IGAB(n_feats, num_blocks=int(igab_blocks[0]), heads=igab_heads)
        self.middle_igab = IGAB(n_feats, num_blocks=int(igab_blocks[1]), heads=igab_heads)
        self.merge_tail = MergeTail(n_feats)
        self.global_illum_head = GlobalIllumHead()
        self.ref_illum = RefIllumTransfer(pool=ref_illum_pool)

        # Debug hooks; no dependence on the logger or legacy trainer.
        self.last_gate = None
        self.last_gate_pre_transform = None
        self.last_stage_modes = None
        self.last_gate_stats = None
        self.last_residual_decomp = None

    @staticmethod
    def _resolve_stage_mode(mode, disable_flag):
        valid = ('normal', 'texture_zero', 'branch_bypass')
        selected = str(mode).lower().strip() if mode not in (None, '') else (
            'texture_zero' if disable_flag else 'normal')
        if selected not in valid:
            raise ValueError('invalid stage mode %s' % selected)
        return selected

    @staticmethod
    def _safe_pad(x, target_hw):
        ph = target_hw[0] - x.shape[-2]
        pw = target_hw[1] - x.shape[-1]
        if ph == 0 and pw == 0:
            return x
        # reflect fails for 1px inputs or if pad >= dimension.
        mode = 'reflect' if ph < x.shape[-2] and pw < x.shape[-1] else 'replicate'
        return F.pad(x, (0, pw, 0, ph), mode=mode)

    @staticmethod
    def _align_ref(T, feat, name):
        if T is None:
            raise ValueError(name + ' required; pass actual SearchTransfer tensors')
        if T.shape[0] != feat.shape[0]:
            raise ValueError(name + ' batch mismatch')
        # SearchTransfer uses floor dimensions; padding Low to a multiple
        # of four may produce up to 3px difference at full resolution.
        # Large mismatches are errors, not automatic global alignment.
        dh = abs(T.shape[-2] - feat.shape[-2])
        dw = abs(T.shape[-1] - feat.shape[-1])
        if dh > 3 or dw > 3:
            raise ValueError(name + ' spatial mismatch greater than 3 pixels: '
                             + str(T.shape[-2:]) + ' vs ' + str(feat.shape[-2:]))
        return resize_like(T, feat)

    @staticmethod
    def _gate(S, feat):
        if S is None:
            raise ValueError('matching score S is required')
        return torch.sigmoid(F.interpolate(S, size=feat.shape[-2:], mode='bicubic',
                                           align_corners=False))

    @staticmethod
    def _apply_gate_mode(g, gate_mode, gate_shift, smooth_kernel):
        if gate_mode in (None, '', 'default', 'a0', 'none'):
            return g
        if gate_mode == 'constant_mean':
            return g.mean(dim=(2, 3), keepdim=True).expand_as(g)
        if gate_mode == 'smooth':
            k = int(smooth_kernel)
            if k <= 0 or k % 2 == 0:
                raise ValueError('gate_smooth_kernel should be positive odd')
            sm = F.avg_pool2d(g, k, stride=1, padding=k // 2)
            return (sm + g.mean(dim=(-2, -1), keepdim=True)
                    - sm.mean(dim=(-2, -1), keepdim=True)).clamp(0, 1)
        if gate_mode == 'shift':
            if gate_shift is None:
                raise ValueError('gate_shift needed for gate_mode=shift')
            if isinstance(gate_shift, str):
                gate_shift = tuple(int(z) for z in gate_shift.split(','))
            return torch.roll(g, shifts=tuple(gate_shift), dims=(-2, -1))
        raise ValueError('unknown gate mode %r' % gate_mode)

    def apply_ref_illum(self, x, low, ref):
        return self.ref_illum(x, low, ref)

    def apply_illum_head(self, x, low):
        gain, bias, gamma = self.global_illum_head(low)
        # Legacy form retained for inference/training protocol compatibility.
        base01 = ((x + 1.) / 2.).clamp(1e-2, 1.)
        y = torch.pow(base01, gamma) * 2. - 1.
        return (y * gain + bias).clamp(-1, 1)

    def _inject(self, feat, T, S, head, mode, gate_mode,
                gate_shift, gate_smooth_kernel):
        if mode == 'branch_bypass':
            return feat
        T = self._align_ref(T, feat, 'T')
        g = self._apply_gate_mode(self._gate(S, feat), gate_mode,
                                  gate_shift, gate_smooth_kernel)
        if mode == 'texture_zero':
            T = torch.zeros_like(T)
        return feat + g * head(torch.cat([feat, T], dim=1))

    @staticmethod
    def _run_rbs(feat, rbs, tail):
        start = feat
        for rb in rbs:
            feat = rb(feat)
        return start + tail(feat)

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None,
                apply_illum=True, ref=None, apply_ref_illum=True,
                gate_mode=None, gate_shift=None, gate_smooth_kernel=31,
                collect_gate_stats=False, disable_T_lv3=False,
                disable_T_lv2=False, disable_T_lv1=False,
                stage_mode_lv3=None, stage_mode_lv2=None, stage_mode_lv1=None,
                collect_residual_decomp=False):
        orig_hw = x.shape[-2:]
        th = int(math.ceil(orig_hw[0] / 4) * 4)
        tw = int(math.ceil(orig_hw[1] / 4) * 4)
        low = self._safe_pad(x, (th, tw))

        modes = [
            self._resolve_stage_mode(stage_mode_lv3, disable_T_lv3),
            self._resolve_stage_mode(stage_mode_lv2, disable_T_lv2),
            self._resolve_stage_mode(stage_mode_lv1, disable_T_lv1),
        ]
        self.last_stage_modes = dict(zip(('lv3', 'lv2', 'lv1'), modes))

        # Retinex light-up always estimates gain in [0,1] RGB coordinates.
        low01 = (low + 1.) * .5
        illu_h, log_gain = self.illumination_estimator(low01)
        if self.enable_lightup:
            # Starting from identity avoids immediate distribution shock to
            # legacy TTSR checkpoint features. Gradients flow through map.
            lightup = low01 * torch.exp(log_gain.clamp(-1.5, 1.5))
            base = 2. * lightup - 1.
        else:
            base = low

        e0 = self.SFE(low)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        illu_h2 = resize_like(illu_h, e1)
        illu_h4 = resize_like(illu_h, e2)

        # Coarse H/4: native reference level lv3.
        x11 = self._inject(e2, T_lv3, S, self.conv11_head, modes[0],
                            gate_mode, gate_shift, gate_smooth_kernel)
        x11 = self._run_rbs(x11, self.RB11, self.conv11_tail)
        x11 = self.coarse_igab(x11, illu_h4)

        # Mid H/2 with encoder skip and bidirectional CSFI2.
        x21 = x11
        x22 = self.skip_e1(torch.cat([resize_like(x11, e1), e1], dim=1))
        x22 = self._inject(x22, T_lv2, S, self.conv22_head, modes[1],
                            gate_mode, gate_shift, gate_smooth_kernel)
        x21, x22 = self.ex12(x21, x22)
        x21 = self._run_rbs(x21, self.RB21, self.conv21_tail)
        x22 = self._run_rbs(x22, self.RB22, self.conv22_tail)
        x22 = self.middle_igab(x22, illu_h2)

        # Fine H with encoder skip and all-scale CSFI3.
        x31, x32 = x21, x22
        x33 = self.skip_e0(torch.cat([resize_like(x22, e0), e0], dim=1))
        x33 = self._inject(x33, T_lv1, S, self.conv33_head, modes[2],
                            gate_mode, gate_shift, gate_smooth_kernel)
        x31, x32, x33 = self.ex123(x31, x32, x33)
        x31 = self._run_rbs(x31, self.RB31, self.conv31_tail)
        x32 = self._run_rbs(x32, self.RB32, self.conv32_tail)
        x33 = self._run_rbs(x33, self.RB33, self.conv33_tail)

        # Residual reconstruction on *illuminated input*, not raw Low.
        rgb_residual = self.merge_tail(x31, x32, x33)
        out = (base + rgb_residual)[..., :orig_hw[0], :orig_hw[1]]
        if apply_ref_illum:
            out = self.apply_ref_illum(out, x, ref)
        if apply_illum:
            out = self.apply_illum_head(out, x)

        # Existing diagnostic scripts may access these attributes. Their old
        # detailed stage residual stats are not implemented for the new arch;
        # never return fabricated comparisons to old same-resolution stages.
        self.last_gate = self._gate(S, e0).detach()
        self.last_gate_pre_transform = self.last_gate
        self.last_gate_stats = None
        self.last_residual_decomp = None
        return out
