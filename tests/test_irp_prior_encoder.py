"""Illumination / Reference prior encoder tests."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.illumination_prior_encoder import (  # noqa: E402
    IlluminationPriorEncoder,
    illumination_prior_map,
)
from asrt.models.reference_prior_encoder import (  # noqa: E402
    ReferencePriorEncoder,
    build_structural_detail_input,
    sobel_magnitude,
)


def test_illum_prior_definition():
    low = torch.zeros(1, 3, 8, 8)
    low[:, 0] = 1.0  # max channel after [0,1] map: (1+1)/2=1 → P=0
    # low in [-1,1]: channel 0 = 1 → x01=1 → P=0
    p = illumination_prior_map(low)
    assert torch.allclose(p, torch.zeros_like(p), atol=1e-5)
    low2 = torch.full((1, 3, 8, 8), -1.0)
    p2 = illumination_prior_map(low2)  # x01=0 → P=1
    assert torch.allclose(p2, torch.ones_like(p2), atol=1e-5)


def test_illum_encoder_scales():
    enc = IlluminationPriorEncoder(64)
    low = torch.randn(2, 3, 64, 64)
    out = enc(low)
    assert out['h1'].shape == (2, 64, 64, 64)
    assert out['h2'].shape == (2, 64, 32, 32)
    assert out['h4'].shape == (2, 64, 16, 16)


def test_illum_only_uses_low():
    enc = IlluminationPriorEncoder(32)
    low = torch.randn(1, 3, 32, 32)
    # no ref argument exists
    out = enc(low)
    assert 'h1' in out


def test_sobel_and_zr():
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    r = torch.rand(1, 3, 32, 32) * 2 - 1
    g = sobel_magnitude((r + 1) * 0.5)
    assert g.shape == (1, 3, 32, 32)
    zr = build_structural_detail_input(x, r, sigma=7.0)
    assert zr.shape == (1, 9, 32, 32)


def test_ref_encoder_scales_and_none():
    enc = ReferencePriorEncoder(64, sigma=7.0)
    low = torch.randn(1, 3, 64, 64)
    ref = torch.randn(1, 3, 64, 64)
    out = enc(low, ref)
    assert out['h1'].shape == (1, 64, 64, 64)
    assert out['h2'].shape == (1, 64, 32, 32)
    assert enc(low, None) is None


def test_same_encoder_class_for_low_and_nano():
    enc = ReferencePriorEncoder(64)
    low = torch.randn(1, 3, 32, 32)
    a = enc(low, low)
    b = enc(low, torch.randn_like(low))
    assert a['h1'].shape == b['h1'].shape


def main():
    test_illum_prior_definition()
    test_illum_encoder_scales()
    test_illum_only_uses_low()
    test_sobel_and_zr()
    test_ref_encoder_scales_and_none()
    test_same_encoder_class_for_low_and_nano()
    print('ALL test_irp_prior_encoder PASS')


if __name__ == '__main__':
    main()
