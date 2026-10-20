"""Reference health diagnostics (not claimed hallucination detectors)."""
from __future__ import annotations

import torch

from asrt.evaluation.metrics import psnr_ssim_rgb


@torch.no_grad()
def compare_ref_modes(model, low, gt, ref, device='cuda'):
    """Correct Nano vs Bypass (ref=None). Optional mismatch if ref provided."""
    model.eval()
    low = low.to(device)
    gt = gt.to(device)
    ref = None if ref is None else ref.to(device)
    y_bypass = model(low, ref=None)
    out = dict(bypass=psnr_ssim_rgb(y_bypass, gt))
    if ref is not None:
        y_correct = model(low, ref=ref)
        out['correct'] = psnr_ssim_rgb(y_correct, gt)
        # cyclic shift mismatch within batch dim-0 only for B=1: flip ref
        y_mis = model(low, ref=torch.flip(ref, dims=[-1]))
        out['mismatch_hflip'] = psnr_ssim_rgb(y_mis, gt)
    return out
