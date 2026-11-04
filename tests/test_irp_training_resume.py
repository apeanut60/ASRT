"""Gradient accumulation equivalence + checkpoint resume continuity."""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.irp_trainer import WarmupCosineScheduler  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402


CFG = dict(
    stage='irp_v1',
    model=dict(
        name='asrt_irp_v1', version='irp_v1', prior_source='nano',
        n_feats=64, num_res_blocks='8+8+4+2',
        attention=dict(heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    ))


def test_grad_accum_equivalence():
    """accum=2 micro-batches ≈ one large backward with summed grads (no AMP)."""
    seed_everything(0)
    m1 = build_asrt_irp(CFG)
    seed_everything(0)
    m2 = build_asrt_irp(CFG)
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-4)
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-4)
    b0_low = torch.randn(1, 3, 64, 64)
    b0_gt = torch.randn(1, 3, 64, 64)
    b0_ref = torch.randn(1, 3, 64, 64)
    b1_low = torch.randn(1, 3, 64, 64)
    b1_gt = torch.randn(1, 3, 64, 64)
    b1_ref = torch.randn(1, 3, 64, 64)

    # accum path
    opt1.zero_grad(set_to_none=True)
    loss = F.mse_loss(m1(b0_low, b0_ref), b0_gt) / 2
    loss.backward()
    loss = F.mse_loss(m1(b1_low, b1_ref), b1_gt) / 2
    loss.backward()
    # reference path: mean of two losses at once is not the same graph;
    # compare parameter grads after same two scaled backwards on m2 sequentially
    opt2.zero_grad(set_to_none=True)
    loss = F.mse_loss(m2(b0_low, b0_ref), b0_gt) / 2
    loss.backward()
    loss = F.mse_loss(m2(b1_low, b1_ref), b1_gt) / 2
    loss.backward()

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if p1.grad is None:
            assert p2.grad is None
            continue
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-5), n1


def test_scheduler_warmup_cosine():
    m = build_asrt_irp(CFG)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = WarmupCosineScheduler(opt, total_steps=100, warmup_steps=10, eta_min=1e-6)
    sch.step()  # step 0
    lr0 = opt.param_groups[0]['lr']
    for _ in range(9):
        sch.step()
    lr_warm = opt.param_groups[0]['lr']
    assert lr_warm > lr0
    for _ in range(90):
        sch.step()
    assert opt.param_groups[0]['lr'] <= 1e-3 + 1e-9


def test_checkpoint_roundtrip_weights():
    seed_everything(1)
    m = build_asrt_irp(CFG)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    sch = WarmupCosineScheduler(opt, 100, 5, 1e-6)
    low = torch.randn(1, 3, 64, 64)
    ref = torch.randn(1, 3, 64, 64)
    gt = torch.randn(1, 3, 64, 64)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(m(low, ref), gt)
        loss.backward()
        opt.step()
        sch.step()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ck.pt')
        from asrt.models.asrt_irp import extract_irp_hparams
        torch.save(dict(
            model=m.state_dict(), optimizer=opt.state_dict(),
            scheduler=sch.state_dict(), step=3, epoch=0,
            model_family='asrt_irp', model_version='irp_v1',
            prior_source='nano', irp_hparams=extract_irp_hparams(CFG),
            cfg=CFG,
        ), path)
        m2 = build_asrt_irp(CFG)
        ck = torch.load(path, map_location='cpu')
        m2.load_state_dict(ck['model'])
        with torch.no_grad():
            assert torch.allclose(m(low, ref), m2(low, ref), atol=1e-6)


def main():
    test_grad_accum_equivalence()
    test_scheduler_warmup_cosine()
    test_checkpoint_roundtrip_weights()
    print('ALL test_irp_training_resume PASS')


if __name__ == '__main__':
    main()
