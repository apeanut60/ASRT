"""Diagnostic hooks must not change default forward math; branch-off restores weights."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402


def _cfg(prior='nano'):
    return dict(model=dict(
        name='asrt_irp_v1', prior_source=prior, n_feats=64,
        num_res_blocks='8+8+4+2', attention=dict(heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    ))


def test_return_aux_matches_default_forward():
    torch.manual_seed(0)
    m = build_asrt_irp(_cfg('nano'))
    m.eval()
    low = torch.randn(1, 3, 64, 64)
    ref = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y0 = m(low, ref)
        y1, aux = m(low, ref, return_aux=True)
        y2 = m(low, ref, collect_branch_stats=True)
    assert torch.allclose(y0, y1, atol=0, rtol=0)
    assert torch.allclose(y0, y2, atol=0, rtol=0)
    assert 'branch_stats' in aux and 'h2' in aux['branch_stats']
    assert torch.isfinite(y0).all()
    for st in aux['branch_stats'].values():
        for k, v in st.items():
            if isinstance(v, float):
                assert v == v  # finite


def test_disable_reference_restores_weights():
    torch.manual_seed(1)
    m = build_asrt_irp(_cfg('nano'))
    sd0 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    low = torch.randn(1, 3, 64, 64)
    ref = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y_on = m(low, ref)
        y_off = m(low, ref, disable_reference=True)
        y_on2 = m(low, ref)
    # weights unchanged
    for k in sd0:
        assert torch.equal(sd0[k], m.state_dict()[k]), k
    # OFF should differ when beta nonzero at init
    assert not torch.equal(y_on, y_off)
    assert torch.equal(y_on, y_on2)


def test_aux_finite_and_ratios():
    m = build_asrt_irp(_cfg('nano'))
    low = torch.randn(2, 3, 48, 48)
    ref = torch.randn(2, 3, 48, 48)
    with torch.no_grad():
        _, aux = m(low, ref, return_aux=True)
    st = aux['branch_stats']['h2']
    assert 0.0 <= st['ref_contribution_ratio'] <= 1.0 + 1e-6
    assert st['beta_ref_mod_l2'] >= 0.0
