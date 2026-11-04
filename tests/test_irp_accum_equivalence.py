"""True grad-accum vs concatenated large-batch equivalence (no AMP)."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402

CFG = dict(model=dict(
    name='asrt_irp_v1', prior_source='nano', n_feats=64,
    num_res_blocks='8+8+4+2',
    attention=dict(heads=4, window_size=8),
    reference_prior=dict(gaussian_sigma=7.0),
    irp=dict(alpha_init=1.0, beta_init=0.1),
))


def test_accum_matches_concat_batch():
    seed_everything(0)
    m_a = build_asrt_irp(CFG)
    seed_everything(0)
    m_b = build_asrt_irp(CFG)

    micros = []
    for i in range(2):
        torch.manual_seed(100 + i)
        low = torch.randn(1, 3, 64, 64)
        ref = torch.randn(1, 3, 64, 64)
        gt = torch.randn(1, 3, 64, 64)
        micros.append((low, ref, gt))

    # Path A: accum
    m_a.zero_grad(set_to_none=True)
    for low, ref, gt in micros:
        loss = F.mse_loss(m_a(low, ref), gt) / 2.0
        loss.backward()

    # Path B: concat large batch
    low_c = torch.cat([x[0] for x in micros], 0)
    ref_c = torch.cat([x[1] for x in micros], 0)
    gt_c = torch.cat([x[2] for x in micros], 0)
    m_b.zero_grad(set_to_none=True)
    loss_b = F.mse_loss(m_b(low_c, ref_c), gt_c)
    loss_b.backward()

    max_grad_diff = 0.0
    for (n1, p1), (n2, p2) in zip(m_a.named_parameters(), m_b.named_parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        d = (p1.grad - p2.grad).abs().max().item()
        max_grad_diff = max(max_grad_diff, d)
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-5), n1

    # Optimizer step equivalence
    opt_a = torch.optim.Adam(m_a.parameters(), lr=1e-4)
    opt_b = torch.optim.Adam(m_b.parameters(), lr=1e-4)
    # re-init opt state identically by stepping once from same grads already present
    opt_a.step()
    opt_b.step()
    max_w = 0.0
    for p1, p2 in zip(m_a.parameters(), m_b.parameters()):
        max_w = max(max_w, float((p1 - p2).abs().max()))
        assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-5)
    print('accum equiv max_grad_diff', max_grad_diff, 'max_w', max_w)


def main():
    test_accum_matches_concat_batch()
    print('ALL test_irp_accum_equivalence PASS')


if __name__ == '__main__':
    main()
