"""RGB PSNR / SSIM in [0,1]. No GT mean-align."""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch


def to_uint8_rgb(t: torch.Tensor, input_range: str = 'minus_one_one') -> np.ndarray:
    """Convert tensor to uint8 HWC.

    Args:
        t: [B,3,H,W] or [3,H,W]
        input_range: 'minus_one_one' (default, ASRT protocol) or 'zero_one'
            Must be specified explicitly — no auto-guess from tensor min.
    """
    if t.dim() == 4:
        t = t[0]
    x = t.detach().float().cpu()
    if input_range == 'minus_one_one':
        x = (x + 1.0) * 0.5
    elif input_range == 'zero_one':
        x = x
    else:
        raise ValueError(
            "input_range must be 'minus_one_one' or 'zero_one', got %r" % (input_range,))
    x = x.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return x


def psnr_rgb(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    diff = (pred_u8.astype(np.float64) - gt_u8.astype(np.float64)) / 255.0
    mse = float(np.mean(diff ** 2))
    if mse <= 0:
        return float('inf')
    return -10.0 * math.log10(mse)


def _ssim_map(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))
    return float(ssim.mean())


def ssim_rgb(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    return float(np.mean([
        _ssim_map(pred_u8[:, :, i], gt_u8[:, :, i]) for i in range(3)]))


def psnr_ssim_rgb(
    pred: torch.Tensor,
    gt: torch.Tensor,
    input_range: str = 'minus_one_one',
) -> dict:
    """Formal ASRT eval: tensors in [-1,1] by default (explicit input_range)."""
    p = to_uint8_rgb(pred, input_range=input_range)
    g = to_uint8_rgb(gt, input_range=input_range)
    if p.shape != g.shape:
        raise ValueError('shape mismatch %s vs %s' % (p.shape, g.shape))
    return dict(psnr_rgb=psnr_rgb(p, g), ssim_rgb=ssim_rgb(p, g))
