"""Wrong mapping + patch/pixel shuffle reference modes."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.reference_modes import (  # noqa: E402
    apply_reference_mode,
    build_wrong_mapping,
    load_wrong_mapping,
    resolve_wrong_ref_id,
    spatial_patch_shuffle,
    spatial_pixel_shuffle,
)


def test_wrong_mapping_no_self_and_bs1():
    ids = ['a', 'b', 'c', 'd']
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'w.json')
        m = build_wrong_mapping(ids, seed=42, out_path=path)
        assert os.path.exists(path)
        m2 = load_wrong_mapping(path)
        assert m == m2
        for k, v in m.items():
            assert k != v
        # BS=1 path: resolve id then apply wrong_ref tensor
        low = torch.randn(1, 3, 32, 32)
        ref = torch.randn(1, 3, 32, 32)
        wid = resolve_wrong_ref_id('a', m)
        wrong = torch.randn(1, 3, 32, 32)
        out = apply_reference_mode(low, ref, mode='wrong', wrong_ref=wrong)
        assert out.shape == ref.shape
        # no flip fallback when mapping/wrong_ref missing
        try:
            apply_reference_mode(low, ref, mode='wrong')
            raise AssertionError('expected failure')
        except RuntimeError:
            pass


def test_patch_vs_pixel_shuffle():
    torch.manual_seed(0)
    ref = torch.randn(1, 3, 64, 64)
    pix = spatial_pixel_shuffle(ref.clone())
    patch = spatial_patch_shuffle(ref.clone(), patch_size=16)
    assert pix.shape == ref.shape and patch.shape == ref.shape
    assert not torch.allclose(pix, ref)
    assert not torch.allclose(patch, ref)
    # patch shuffle preserves per-patch content set (sum approx over patches)
    # at least distinct from pixel shuffle typically
    assert not torch.allclose(pix, patch, atol=1e-5)


def test_modes_only_change_ref_branch_input():
    low = torch.randn(1, 3, 32, 32)
    ref = torch.randn(1, 3, 32, 32)
    c = apply_reference_mode(low, ref, 'correct')
    n = apply_reference_mode(low, ref, 'noref')
    rl = apply_reference_mode(low, ref, 'ref_low')
    assert torch.allclose(c, ref)
    assert n is None
    assert torch.allclose(rl, low)


def main():
    test_wrong_mapping_no_self_and_bs1()
    test_patch_vs_pixel_shuffle()
    test_modes_only_change_ref_branch_input()
    print('ALL test_irp_reference_modes PASS')


if __name__ == '__main__':
    main()
