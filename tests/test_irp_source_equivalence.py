"""A1/A2/A3 architecture consistency and source equivalence."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt import ASRT  # noqa: E402
from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.models.factory import build_model  # noqa: E402


def _cfg(src):
    return dict(model=dict(
        name='asrt_irp_v1', n_feats=64, num_res_blocks='8+8+4+2',
        prior_source=src, attention=dict(heads=4, window_size=8),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    ))


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


def test_a2_a3_param_count_equal():
    a2 = build_asrt_irp(_cfg('low'))
    a3 = build_asrt_irp(_cfg('nano'))
    assert _nparams(a2) == _nparams(a3)


def test_a3_ref_eq_low_matches_a2():
    a2 = build_asrt_irp(_cfg('low'))
    a3 = build_asrt_irp(_cfg('nano'))
    a3.load_state_dict(a2.state_dict())
    low = torch.randn(1, 3, 64, 64)
    a2.eval()
    a3.eval()
    with torch.no_grad():
        y2 = a2(low, None)
        y3 = a3(low, low)
    assert torch.allclose(y2, y3, atol=1e-5)


def test_a1_no_ref_branch_params_lower():
    a1 = build_asrt_irp(_cfg('none'))
    a2 = build_asrt_irp(_cfg('low'))
    assert _nparams(a1) < _nparams(a2)
    # A1 should not expose ref attention modules when use_reference=False
    assert a1.backbone.middle_irp.attn_ref is None or not a1.backbone.middle_irp.use_reference
    assert a1.backbone.fine_irp.attn_ref is None or not a1.backbone.fine_irp.use_reference


def test_legacy_asrt_still_loads():
    legacy_cfg = dict(model=dict(
        name='asrt', n_feats=64, num_res_blocks='8+8+4+2',
    ))
    try:
        m = build_model(legacy_cfg)
    except (KeyError, ValueError, TypeError, AttributeError):
        # Factory may not register legacy name; fall back to direct ctor.
        m = ASRT(n_feats=64, num_res_blocks='8+8+4+2')
    low = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = m(low)
    assert y.shape == low.shape


def test_a2_a3_same_training_api():
    for src in ('low', 'nano'):
        m = build_asrt_irp(_cfg(src))
        low = torch.randn(2, 3, 64, 64)
        ref = torch.randn(2, 3, 64, 64) if src == 'nano' else None
        gt = torch.randn(2, 3, 64, 64)
        pred = m(low, ref)
        loss = torch.nn.functional.mse_loss(pred, gt)
        loss.backward()
        assert torch.isfinite(loss)


def main():
    test_a2_a3_param_count_equal()
    test_a3_ref_eq_low_matches_a2()
    test_a1_no_ref_branch_params_lower()
    test_legacy_asrt_still_loads()
    test_a2_a3_same_training_api()
    print('ALL test_irp_source_equivalence PASS')


if __name__ == '__main__':
    main()
