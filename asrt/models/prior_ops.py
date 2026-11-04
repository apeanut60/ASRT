"""Model-local prior operators (Gaussian / high-pass / Sobel helpers).

Ported from the historical R1.x `reference_information` diagnostics module so
IRP does not depend on evaluation-only scripts that may be absent from a
clean checkout of this branch.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def gaussian_radius(sigma: float, truncate: float = 3.0) -> int:
    """Half-width of the 1D Gaussian kernel (truncate * sigma)."""
    return max(1, int(math.ceil(float(truncate) * float(sigma))))


def gaussian_kernel1d(
    sigma: float,
    device,
    dtype,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Normalized 1D Gaussian kernel. Length = 2*radius+1."""
    sigma = float(sigma)
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = gaussian_radius(sigma, truncate=truncate)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum().clamp(min=1e-12)


def gaussian_blur(
    x: torch.Tensor,
    sigma: float,
    truncate: float = 3.0,
    padding_mode: str = 'reflect',
) -> torch.Tensor:
    """Separable Gaussian blur for [B,C,H,W].

    Kernel size = 2*ceil(truncate*sigma)+1 (shrunk if image is smaller).
    Padding: reflect (falls back to replicate if reflect is unsafe).
    Autograd-safe; works on CPU/CUDA and under AMP (uses input dtype).
    """
    if sigma is None or float(sigma) <= 0:
        return x
    if x.dim() != 4:
        raise ValueError('gaussian_blur expects [B,C,H,W], got %s' % (tuple(x.shape),))
    b, c, h, w = x.shape
    if h < 1 or w < 1:
        return x
    k1 = gaussian_kernel1d(sigma, x.device, x.dtype, truncate=truncate)
    radius = (k1.numel() - 1) // 2
    r = min(radius, max(0, h - 1), max(0, w - 1))
    if r < 1:
        return x
    if r != radius:
        mid = k1.numel() // 2
        k1 = k1[mid - r: mid + r + 1]
        k1 = k1 / k1.sum().clamp(min=1e-12)
        radius = r
    ky = k1.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
    kx = k1.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
    mode = padding_mode
    # reflect requires pad < spatial size
    if mode == 'reflect' and (radius >= h or radius >= w):
        mode = 'replicate'
    x = F.pad(x, (radius, radius, radius, radius), mode=mode)
    x = F.conv2d(x, ky, groups=c)
    x = F.conv2d(x, kx, groups=c)
    return x


def high_pass(
    x: torch.Tensor,
    sigma: float = 7.0,
    truncate: float = 3.0,
) -> torch.Tensor:
    """x - gaussian_blur(x, sigma)."""
    return x - gaussian_blur(x, sigma=sigma, truncate=truncate)


def prior_context_halo(sigma: float = 7.0, truncate: float = 3.0, sobel: int = 1) -> int:
    """Minimum spatial context (pixels) needed so center HP/Sobel matches full-image."""
    return int(gaussian_radius(sigma, truncate=truncate) + int(sobel))


def kernel_meta(sigma: float = 7.0, truncate: float = 3.0) -> dict:
    r = gaussian_radius(sigma, truncate=truncate)
    return dict(
        sigma=float(sigma),
        truncate=float(truncate),
        radius=r,
        kernel_size=2 * r + 1,
        padding_mode='reflect',
        separable=True,
    )
