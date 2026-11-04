"""Dual-Prior Windowed Self-Attention (IRP-MSA) for ASRT vNext V1.

Q/K/V from restoration features only; priors modulate attention outputs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def pad_to_window(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad NCHW to multiples of window_size. Returns padded, (pad_b, pad_r, H, W)."""
    _, _, h, w = x.shape
    ph = (window_size - h % window_size) % window_size
    pw = (window_size - w % window_size) % window_size
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='constant', value=0.0)
    return x, (ph, pw, h, w)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """B,C,H,W → [B*nW, ws*ws, C]. H,W must be divisible by ws."""
    b, c, h, w = x.shape
    ws = window_size
    x = x.view(b, c, h // ws, ws, w // ws, ws)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return windows.view(-1, ws * ws, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    """[B*nW, ws*ws, C] → B,C,H,W."""
    ws = window_size
    c = windows.shape[-1]
    x = windows.view(b, h // ws, w // ws, ws, ws, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, h, w)


def build_window_key_mask(
    h: int, w: int, ph: int, pw: int, window_size: int, device, dtype
) -> Optional[torch.Tensor]:
    """Bool mask True where key is padding. Shape [nW, 1, 1, ws*ws] broadcastable to attn.

    Returns None if no padding.
    """
    if ph == 0 and pw == 0:
        return None
    ws = window_size
    hp, wp = h + ph, w + pw
    valid = torch.zeros(hp, wp, device=device, dtype=torch.bool)
    valid[:h, :w] = True
    # partition like windows: [nWh, nWw, ws, ws]
    n_h, n_w = hp // ws, wp // ws
    v = valid.view(n_h, ws, n_w, ws).permute(0, 2, 1, 3).contiguous()
    v = v.view(n_h * n_w, ws * ws)  # True = valid
    # attn mask: True means mask out (padding keys)
    key_pad = ~v  # [nW, tokens]
    return key_pad.unsqueeze(1).unsqueeze(2)  # [nW,1,1,tokens]


class WindowSelfAttention(nn.Module):
    """Independent windowed MHSA; Q/K/V from feature only."""

    def __init__(self, dim: int = 64, heads: int = 4, window_size: int = 8):
        super().__init__()
        if dim % heads != 0:
            raise ValueError('dim must be divisible by heads')
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = int(window_size)
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,C,H,W] → [B,C,H,W]."""
        b, c, h, w = x.shape
        ws = self.window_size
        xp, (ph, pw, oh, ow) = pad_to_window(x, ws)
        hp, wp = xp.shape[-2:]
        mask = build_window_key_mask(oh, ow, ph, pw, ws, x.device, x.dtype)

        windows = window_partition(xp, ws)  # [B*nW, N, C]
        n_win = windows.shape[0] // b
        q = self.q(windows)
        k = self.k(windows)
        v = self.v(windows)

        def reshape_heads(t):
            return t.view(-1, t.shape[1], self.heads, self.head_dim).permute(0, 2, 1, 3)

        qh, kh, vh = reshape_heads(q), reshape_heads(k), reshape_heads(v)
        attn = (qh * self.scale) @ kh.transpose(-2, -1)  # [B*nW, heads, N, N]
        if mask is not None:
            # mask: [nW,1,1,N] → expand over batch of windows
            m = mask.repeat(b, 1, 1, 1)  # [B*nW,1,1,N]
            attn = attn.masked_fill(m, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # all-masked rows → 0
        out = (attn @ vh).permute(0, 2, 1, 3).contiguous().view(-1, ws * ws, c)
        out = self.proj(out)
        out = window_reverse(out, ws, hp, wp, b)
        if ph or pw:
            out = out[:, :, :oh, :ow]
        return out


class ConvFFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class IRPTransformerBlock(nn.Module):
    """Illumination ± Reference prior-modulated dual self-attention block."""

    def __init__(
        self,
        dim: int = 64,
        heads: int = 4,
        window_size: int = 8,
        use_reference: bool = True,
        alpha_init: float = 1.0,
        beta_init: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.use_reference = bool(use_reference)
        self.norm1 = LayerNorm2d(dim)
        self.attn_illum = WindowSelfAttention(dim, heads, window_size)
        self.illum_projection = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        if self.use_reference:
            self.attn_ref = WindowSelfAttention(dim, heads, window_size)
            self.ref_projection = nn.Conv2d(dim, dim, 1)
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        else:
            self.attn_ref = None
            self.ref_projection = None
            self.beta = None
        self.output_projection = nn.Conv2d(dim, dim, 1)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = ConvFFN(dim)

    def forward(
        self,
        feature: torch.Tensor,
        illumination_prior: torch.Tensor,
        reference_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if illumination_prior.shape[-2:] != feature.shape[-2:]:
            illumination_prior = F.interpolate(
                illumination_prior, size=feature.shape[-2:],
                mode='bilinear', align_corners=False)
        f = self.norm1(feature)
        illum_feat = self.attn_illum(f)
        illum_mod = illum_feat * self.illum_projection(illumination_prior)
        fused = self.alpha * illum_mod

        if self.use_reference and reference_prior is not None and self.attn_ref is not None:
            if reference_prior.shape[-2:] != feature.shape[-2:]:
                reference_prior = F.interpolate(
                    reference_prior, size=feature.shape[-2:],
                    mode='bilinear', align_corners=False)
            ref_feat = self.attn_ref(f)
            ref_mod = ref_feat * torch.tanh(self.ref_projection(reference_prior))
            fused = fused + self.beta * ref_mod

        x = feature + self.output_projection(fused)
        x = x + self.ffn(self.norm2(x))
        return x
