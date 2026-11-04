"""Gradient flow through IRP modules."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402

CFG = dict(model=dict(
    name='asrt_irp_v1', n_feats=64, num_res_blocks='8+8+4+2', prior_source='nano',
    attention=dict(heads=4, window_size=8),
    irp=dict(alpha_init=1.0, beta_init=0.1),
))


def _has_grad(module):
    return any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in module.parameters() if p.requires_grad)


def test_gradients():
    m = build_asrt_irp(CFG)
    low = torch.randn(1, 3, 64, 64)
    ref = torch.randn(1, 3, 64, 64)
    gt = torch.randn(1, 3, 64, 64)
    pred = m(low, ref)
    loss = F.mse_loss(pred, gt)
    loss.backward()
    bb = m.backbone
    assert _has_grad(bb.coarse_irp.attn_illum)
    assert _has_grad(bb.middle_irp.attn_illum)
    assert _has_grad(bb.middle_irp.attn_ref)
    assert _has_grad(bb.fine_irp.attn_ref)
    assert _has_grad(bb.illum_prior_encoder)
    assert _has_grad(bb.ref_prior_encoder)
    assert bb.middle_irp.alpha.grad is not None
    assert bb.middle_irp.beta.grad is not None
    assert _has_grad(bb.SFE)
    assert _has_grad(bb.merge_tail)
    print('gradient checks OK')


def main():
    test_gradients()
    print('ALL test_irp_gradient_flow PASS')


if __name__ == '__main__':
    main()
