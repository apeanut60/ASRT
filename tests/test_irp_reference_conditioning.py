"""Reference conditioning / NoRef behavior."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.models.reference_prior_encoder import build_structural_detail_input  # noqa: E402


def _cfg(src):
    return dict(model=dict(
        name='asrt_irp_v1', n_feats=64, num_res_blocks='8+8+4+2',
        prior_source=src, attention=dict(heads=4, window_size=8),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    ))


def test_wrong_ref_changes_prior_and_output():
    low = torch.randn(1, 3, 64, 64)
    r1 = torch.randn(1, 3, 64, 64)
    r2 = torch.randn(1, 3, 64, 64)
    z1 = build_structural_detail_input(low, r1)
    z2 = build_structural_detail_input(low, r2)
    assert not torch.allclose(z1, z2, atol=1e-5)
    m = build_asrt_irp(_cfg('nano'))
    m.eval()
    with torch.no_grad():
        y1 = m(low, r1)
        y2 = m(low, r2)
    # At random init beta=0.1, RGB delta can be ~1e-6; still must be nonzero.
    assert float((y1 - y2).abs().max()) > 1e-8


def test_noref_skips_ref_encoder():
    m = build_asrt_irp(_cfg('nano'))
    called = {'n': 0}
    orig = m.backbone.ref_prior_encoder.forward

    def wrapped(*a, **k):
        called['n'] += 1
        return orig(*a, **k)

    m.backbone.ref_prior_encoder.forward = wrapped
    low = torch.randn(1, 3, 64, 64)
    _ = m(low, None)
    assert called['n'] == 0
    _ = m(low, torch.randn_like(low))
    assert called['n'] == 1


def test_a2_ignores_external_ref():
    m = build_asrt_irp(_cfg('low'))
    m.eval()
    low = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y1 = m(low, torch.randn_like(low))
        y2 = m(low, torch.randn_like(low))
        y0 = m(low, None)
    assert torch.allclose(y1, y2, atol=1e-6)
    assert torch.allclose(y1, y0, atol=1e-6)


def main():
    test_wrong_ref_changes_prior_and_output()
    test_noref_skips_ref_encoder()
    test_a2_ignores_external_ref()
    print('ALL test_irp_reference_conditioning PASS')


if __name__ == '__main__':
    main()
