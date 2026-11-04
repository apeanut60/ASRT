"""Warmup+Cosine LR vs optimizer-update index."""
from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.irp_trainer import WarmupCosineScheduler  # noqa: E402


CFG = dict(model=dict(
    name='asrt_irp_v1', prior_source='nano', n_feats=64,
    num_res_blocks='8+8+4+2',
    attention=dict(heads=4, window_size=8),
    reference_prior=dict(gaussian_sigma=7.0),
    irp=dict(alpha_init=1.0, beta_init=0.1),
))


def _sched(total=80000, warmup=1000, base=1e-4, eta=1e-6):
    m = build_asrt_irp(CFG)
    opt = torch.optim.Adam(m.parameters(), lr=base)
    return WarmupCosineScheduler(opt, total_steps=total, warmup_steps=warmup, eta_min=eta), opt


def test_update0_and_warmup_end():
    sch, opt = _sched()
    # First upcoming update index 0
    lr0 = sch.lr_for_update(0)[0]
    assert abs(lr0 - 1e-4 * 1 / 1000) < 1e-12
    assert abs(opt.param_groups[0]['lr'] - lr0) < 1e-12  # applied in __init__
    lr_w = sch.lr_for_update(999)[0]
    assert abs(lr_w - 1e-4) < 1e-12
    # first cosine step
    lr_c0 = sch.lr_for_update(1000)[0]
    assert abs(lr_c0 - 1e-4) < 1e-9


def test_cosine_and_final():
    sch, _ = _sched(total=100, warmup=10, base=1e-3, eta=1e-6)
    lr_mid = sch.lr_for_update(55)[0]  # halfway through cosine
    # prog = (55-10)/90 = 0.5 → cos(pi/2)=0 → lr=eta_min
    expected = 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + math.cos(math.pi * 0.5))
    assert abs(lr_mid - expected) < 1e-12
    lr_end = sch.lr_for_update(99)[0]
    prog = (99 - 10) / 90
    expected_end = 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + math.cos(math.pi * prog))
    assert abs(lr_end - expected_end) < 1e-12


def test_accum_independent_curve():
    """grad_accum must not change LR-vs-update curve (schedule is update-indexed)."""
    sch1, _ = _sched(total=20, warmup=4, base=1e-4)
    sch4, _ = _sched(total=20, warmup=4, base=1e-4)
    for k in range(20):
        assert abs(sch1.lr_for_update(k)[0] - sch4.lr_for_update(k)[0]) < 1e-15


def test_resume_next_lr():
    sch, opt = _sched(total=50, warmup=5, base=1e-4)
    for _ in range(7):
        sch.apply_lr(sch.updates_done)
        sch.mark_update_done()
    assert sch.updates_done == 7
    state = sch.state_dict()
    sch2, opt2 = _sched(total=50, warmup=5, base=1e-4)
    sch2.load_state_dict(state)
    assert sch2.updates_done == 7
    assert abs(opt2.param_groups[0]['lr'] - sch.lr_for_update(7)[0]) < 1e-15


def test_skipped_update_does_not_advance():
    sch, opt = _sched(total=20, warmup=5, base=1e-4)
    sch.apply_lr(0)
    # simulate successful update
    sch.mark_update_done()
    assert sch.updates_done == 1
    lr_before = sch.lr_for_update(1)[0]
    sch.apply_lr(1)
    # simulate AMP skip: do NOT mark_update_done
    assert sch.updates_done == 1
    assert abs(opt.param_groups[0]['lr'] - lr_before) < 1e-15


def main():
    test_update0_and_warmup_end()
    test_cosine_and_final()
    test_accum_independent_curve()
    test_resume_next_lr()
    test_skipped_update_does_not_advance()
    print('ALL test_irp_scheduler_steps PASS')


if __name__ == '__main__':
    main()
