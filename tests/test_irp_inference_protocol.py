"""IRP full/tiled inference protocol tests."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.evaluation.inference import (  # noqa: E402
    call_model,
    full_vs_tiled_stats,
    infer_full,
    infer_tiled,
    is_irp_model,
)
from asrt.evaluation.protocol import infer_with_protocol  # noqa: E402
from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.models.factory import build_model  # noqa: E402


CFG = dict(model=dict(
    name='asrt_irp_v1', n_feats=64, num_res_blocks='8+8+4+2',
    prior_source='nano', attention=dict(heads=4, window_size=8),
    irp=dict(alpha_init=1.0, beta_init=0.1),
    reference_prior=dict(gaussian_sigma=7.0),
))


def test_full_and_tiled_shapes():
    m = build_asrt_irp(CFG)
    m.eval()
    assert is_irp_model(m)
    for h, w in ((128, 128), (160, 192), (100, 120), (64, 80)):
        low = torch.randn(1, 3, h, w)
        ref = torch.randn(1, 3, h, w)
        y = infer_full(m, low, ref)
        assert y.shape == low.shape
        yt, wt = infer_tiled(m, low, ref, tile=64, overlap=16, return_weight=True)
        assert yt.shape == low.shape
        assert int((wt <= 0).sum()) == 0
        # NoRef
        yn = infer_full(m, low, None)
        assert yn.shape == low.shape


def test_large_and_unequal():
    m = build_asrt_irp(CFG).eval()
    low = torch.randn(1, 3, 520, 480)
    ref = torch.randn_like(low)
    y = infer_full(m, low, ref)
    assert y.shape == low.shape
    stats = full_vs_tiled_stats(m, low[:, :, :256, :240], ref[:, :, :256, :240],
                                tile=128, overlap=32)
    assert stats['uncovered'] == 0
    assert stats['is_irp'] is True
    print('full_vs_tiled', stats)


def test_protocol_forces_full_for_irp():
    m = build_asrt_irp(CFG).eval()
    low = torch.randn(1, 3, 64, 64)
    # tiled without allow → error when mode=tiled
    try:
        infer_with_protocol(
            m, low, None,
            eval_cfg={'evaluation': {'inference_mode': 'tiled', 'irp_allow_tiled': False}})
        raise AssertionError('expected RuntimeError')
    except RuntimeError as e:
        assert 'refuses unverified tiled' in str(e)
    # with allow
    pred, meta = infer_with_protocol(
        m, low, None,
        eval_cfg={'evaluation': {
            'inference_mode': 'tiled', 'irp_allow_tiled': True,
            'tile': 64, 'overlap': 16}})
    assert meta['inference_used'] == 'tiled'
    assert pred.shape == low.shape
    # default full
    pred2, meta2 = infer_with_protocol(
        m, low, None,
        eval_cfg={'evaluation': {'inference_mode': 'full'}})
    assert meta2['inference_used'] == 'full'


def test_call_model_no_query_kwargs_for_irp():
    m = build_asrt_irp(CFG).eval()
    low = torch.randn(1, 3, 32, 32)
    y = call_model(m, low, ref=None, query_gamma=torch.ones(1), query_lut=None)
    assert y.shape == low.shape


def test_legacy_regression():
    m = build_model({'stage': 'a', 'model': {'n_feats': 64, 'num_res_blocks': '8+8+4+2'}})
    m.eval()
    low = torch.randn(1, 3, 64, 64)
    y = infer_full(m, low, None)
    assert y.shape == low.shape
    yt = infer_tiled(m, low, None, tile=64, overlap=16)
    assert yt.shape == low.shape


def main():
    test_full_and_tiled_shapes()
    test_large_and_unequal()
    test_protocol_forces_full_for_irp()
    test_call_model_no_query_kwargs_for_irp()
    test_legacy_regression()
    print('ALL test_irp_inference_protocol PASS')


if __name__ == '__main__':
    main()
