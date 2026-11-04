"""IRP model shape / AMP / old ASRT compatibility."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.models.factory import build_model  # noqa: E402


CFG = dict(
    model=dict(
        name='asrt_irp_v1', n_feats=64, num_res_blocks='8+8+4+2',
        prior_source='nano',
        attention=dict(heads=4, window_size=8),
        irp=dict(alpha_init=1.0, beta_init=0.1),
        reference_prior=dict(gaussian_sigma=7.0),
    ))


def _run(h, w, prior='nano'):
    c = dict(CFG); c['model'] = dict(CFG['model']); c['model']['prior_source'] = prior
    m = build_asrt_irp(c)
    low = torch.randn(2, 3, h, w)
    ref = torch.randn(2, 3, h, w) if prior == 'nano' else None
    y = m(low, ref)
    assert y.shape == (2, 3, h, w)
    assert torch.isfinite(y).all()
    return y


def test_shapes():
    _run(128, 128, 'nano')
    _run(256, 256, 'nano')
    _run(100, 120, 'nano')  # non-multiple of 4 handled by pad
    _run(128, 128, 'none')
    _run(128, 128, 'low')


def test_amp():
    if not torch.cuda.is_available():
        print('AMP skip (no cuda)')
        return
    m = build_asrt_irp(CFG).cuda()
    low = torch.randn(1, 3, 128, 128, device='cuda')
    ref = torch.randn(1, 3, 128, 128, device='cuda')
    with torch.cuda.amp.autocast():
        y = m(low, ref)
    assert torch.isfinite(y.float()).all()


def test_old_asrt_still_works():
    m = build_model({'stage': 'a', 'model': {'n_feats': 64, 'num_res_blocks': '8+8+4+2'}})
    y = m(torch.randn(1, 3, 64, 64), None)
    assert y.shape == (1, 3, 64, 64)


def main():
    test_shapes()
    test_amp()
    test_old_asrt_still_works()
    print('ALL test_irp_model_shapes PASS')


if __name__ == '__main__':
    main()
