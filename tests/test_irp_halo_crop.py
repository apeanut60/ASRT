"""Halo crop vs full-image high-pass consistency."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.halo_crop import compute_irp_halo, sync_geometry_halo  # noqa: E402
from asrt.models.prior_ops import high_pass, prior_context_halo  # noqa: E402
from asrt.models.reference_prior_encoder import sobel_magnitude  # noqa: E402


def test_halo_size_matches_kernel():
    h = compute_irp_halo(sigma=7.0, window_size=8)
    assert h >= prior_context_halo(7.0, sobel=1)


def test_center_hp_matches_full():
    torch.manual_seed(0)
    # Full image
    img = torch.rand(1, 3, 256, 256)
    sigma = 7.0
    halo = prior_context_halo(sigma, sobel=1)
    crop = 128
    top, left = 40, 50
    full_hp = high_pass(img, sigma=sigma)
    full_center = full_hp[:, :, top:top + crop, left:left + crop]

    # Halo crop then HP then center
    y0, x0 = top - halo, left - halo
    patch = img[:, :, y0:y0 + crop + 2 * halo, x0:x0 + crop + 2 * halo]
    patch_hp = high_pass(patch, sigma=sigma)
    patch_center = patch_hp[:, :, halo:halo + crop, halo:halo + crop]
    err = (full_center - patch_center).abs()
    # Reflect padding at image border can differ if crop near edge; interior OK
    assert float(err.max()) < 1e-4, err.max()


def test_sobel_center_matches():
    img = torch.rand(1, 3, 200, 200)
    halo = 1
    crop = 64
    top, left = 30, 30
    full = sobel_magnitude(img)
    full_c = full[:, :, top:top + crop, left:left + crop]
    patch = img[:, :, top - halo:top + crop + halo, left - halo:left + crop + halo]
    pc = sobel_magnitude(patch)[:, :, halo:halo + crop, halo:halo + crop]
    assert float((full_c - pc).abs().max()) < 1e-4


def test_sync_geometry_halo_shapes():
    rng = np.random.RandomState(0)
    t = torch.rand(3, 300, 300)
    packed = sync_geometry_halo([t, t.clone()], crop=128, rng=rng, halo=22, window_size=8)
    assert packed['tensors_halo'][0].shape[-1] == 128 + 44
    assert packed['tensors_valid'][0].shape[-1] == 128


def main():
    test_halo_size_matches_kernel()
    test_center_hp_matches_full()
    test_sobel_center_matches()
    test_sync_geometry_halo_shapes()
    print('ALL test_irp_halo_crop PASS')


if __name__ == '__main__':
    main()
