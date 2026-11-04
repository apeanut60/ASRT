"""Halo crop / multi-scale window-grid alignment tests."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.halo_crop import (  # noqa: E402
    compute_irp_halo,
    sync_geometry_halo,
    window_grid_alignment,
)
from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.models.irp_attention import window_partition  # noqa: E402
from asrt.models.prior_ops import high_pass, prior_context_halo  # noqa: E402
from asrt.models.reference_prior_encoder import sobel_magnitude  # noqa: E402


def test_halo_rounds_to_alignment():
    h = compute_irp_halo(sigma=7.0, window_size=8)
    align = window_grid_alignment(8)
    assert align == 32
    assert h >= prior_context_halo(7.0, sobel=1)
    assert h % align == 0
    assert h == 32  # ceil(22/32)*32


def test_center_hp_matches_full():
    torch.manual_seed(0)
    img = torch.rand(1, 3, 256, 256)
    sigma = 7.0
    halo = prior_context_halo(sigma, sobel=1)
    crop = 128
    top, left = 40, 50
    full_hp = high_pass(img, sigma=sigma)
    full_center = full_hp[:, :, top:top + crop, left:left + crop]
    y0, x0 = top - halo, left - halo
    patch = img[:, :, y0:y0 + crop + 2 * halo, x0:x0 + crop + 2 * halo]
    patch_hp = high_pass(patch, sigma=sigma)
    patch_center = patch_hp[:, :, halo:halo + crop, halo:halo + crop]
    err = (full_center - patch_center).abs()
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


def test_outer_and_valid_alignment():
    ws = 8
    align = window_grid_alignment(ws)
    halo = compute_irp_halo(sigma=7.0, window_size=ws)
    assert halo % align == 0
    rng = np.random.RandomState(1)
    t = torch.rand(3, 400, 400)
    for _ in range(30):
        packed = sync_geometry_halo(
            [t], crop=128, rng=rng, halo=halo, window_size=ws,
            align_valid_to_window=True)
        meta = packed['meta']
        assert meta['outer_top'] % align == 0, meta
        assert meta['outer_left'] % align == 0, meta
        assert meta['valid_top'] == meta['outer_top'] + halo
        assert meta['valid_left'] == meta['outer_left'] + halo
        assert meta['valid_top'] % align == 0
        assert meta['valid_left'] % align == 0
        assert meta['outer_aligned'] is True
        assert packed['tensors_halo'][0].shape[-1] == 128 + 2 * halo
        assert packed['tensors_valid'][0].shape[-1] == 128


def test_window_token_global_coords():
    """Outer-aligned crop: local window origins map to global multiples of align."""
    ws = 8
    align = window_grid_alignment(ws)
    halo = compute_irp_halo(7.0, ws)
    rng = np.random.RandomState(2)
    H = 320
    t = torch.rand(3, H, H)
    packed = sync_geometry_halo(
        [t], crop=128, rng=rng, halo=halo, window_size=ws)
    meta = packed['meta']
    ot, ol = meta['outer_top'], meta['outer_left']
    # Local window (wy, wx) covers local pixels [wy*ws, (wy+1)*ws)
    # Global = local + outer
    for wy in range(0, 128 // ws):
        for wx in range(0, 128 // ws):
            gy = ot + wy * ws
            gx = ol + wx * ws
            assert gy % ws == 0
            assert gx % ws == 0
            # With outer % 32 == 0: gy % 8 always; %16 when wy even; %32 when wy%4==0
            if wy % 2 == 0 and wx % 2 == 0:
                assert gy % (ws * 2) == 0
                assert gx % (ws * 2) == 0
            if wy % 4 == 0 and wx % 4 == 0:
                assert gy % align == 0
                assert gx % align == 0


def test_sync_geometry_flip_rot_same_for_all():
    rng = np.random.RandomState(0)
    a = torch.rand(3, 256, 256)
    b = a.clone()
    c = a.clone()
    packed = sync_geometry_halo(
        [a, b, c], crop=64, rng=rng, halo=32, window_size=8)
    meta = packed['meta']
    assert meta['flip_w'] in (True, False)
    ha = packed['tensors_halo']
    assert torch.equal(ha[0], ha[1]) and torch.equal(ha[1], ha[2])


def test_non_multiple_and_border():
    rng = np.random.RandomState(3)
    # Non-multiple size; near-border via small image forcing pad
    t = torch.rand(3, 150, 173)
    packed = sync_geometry_halo(
        [t], crop=64, rng=rng, halo=32, window_size=8)
    assert packed['tensors_valid'][0].shape[-2:] == (64, 64)
    assert torch.isfinite(packed['tensors_halo'][0]).all()


def test_halo_forward_backward():
    m = build_asrt_irp(dict(model=dict(
        name='asrt_irp_v1', prior_source='nano', n_feats=64,
        num_res_blocks='8+8+4+2', attention=dict(heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    )))
    halo = compute_irp_halo(7.0, 8)
    # model sees halo-inclusive crop size
    s = 64 + 2 * halo
    low = torch.randn(1, 3, s, s)
    ref = torch.randn(1, 3, s, s)
    y = m(low, ref)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert any(p.grad is not None for p in m.parameters())


def test_a2_a3_same_halo():
    h2 = compute_irp_halo(7.0, 8)
    h3 = compute_irp_halo(7.0, 8)
    assert h2 == h3 == 32


def main():
    test_halo_rounds_to_alignment()
    test_center_hp_matches_full()
    test_sobel_center_matches()
    test_outer_and_valid_alignment()
    test_window_token_global_coords()
    test_sync_geometry_flip_rot_same_for_all()
    test_non_multiple_and_border()
    test_halo_forward_backward()
    test_a2_a3_same_halo()
    print('ALL test_irp_halo_crop PASS')


if __name__ == '__main__':
    main()
