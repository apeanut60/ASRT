"""Interrupted vs uninterrupted IRP training equivalence (fail-fast)."""
from __future__ import annotations

import copy
import os
import sys
import tempfile

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.training.irp_trainer import IRPTrainer  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402
from tests.irp_test_fixtures import apply_synth_paths, make_synthetic_triplet_dataset  # noqa: E402

# CUDA AMP soft threshold — exceed ⇒ FAIL (not PASS_SOFT)
CUDA_AMP_PARAM_ATOL = 1e-3
CPU_PARAM_ATOL = 1e-6
CPU_LOSS_ATOL = 1e-6


def _cfg(tmp, synth, *, accum=1, amp=False, num_workers=0, max_steps=6, device='cpu'):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1.yaml')))
    cfg = apply_synth_paths(cfg, synth)
    cfg['out_dir'] = tmp
    cfg['model']['prior_source'] = 'nano'
    cfg['data']['ref_mode'] = 'correct'
    cfg['training'].update(dict(
        max_steps=max_steps, batch_size=2, num_workers=num_workers, amp=amp,
        grad_accum=accum, warmup_steps=2, val_every=10000, ckpt_every=10000,
        log_every=1, device=device, halo=0, crop_size=64, seed=42,
        optimizer='adam', learning_rate=1e-4, drop_last=True,
    ))
    cfg['seed'] = 42
    cfg['evaluation'] = dict(
        inference_mode='full', irp_allow_tiled=False, irp_oom_policy='error',
        input_range='minus_one_one', tile=256, overlap=32, full_max_side=100000,
    )
    return cfg


def _run_uninterrupted(cfg):
    seed_everything(42)
    tr = IRPTrainer(cfg, device=cfg['training']['device'])
    logs = []
    while tr.global_step < tr.max_steps:
        logs.append(tr.train_update())
    return tr, logs


def _run_interrupted(cfg, split_at=3):
    seed_everything(42)
    cfg_a = copy.deepcopy(cfg)
    tr1 = IRPTrainer(cfg_a, device=cfg['training']['device'])
    logs1 = []
    while tr1.global_step < split_at:
        logs1.append(tr1.train_update())
    ck = tr1.save_checkpoint('interrupt')
    del tr1
    cfg_b = copy.deepcopy(cfg)
    cfg_b['out_dir'] = cfg['out_dir'] + '_resume'
    os.makedirs(cfg_b['out_dir'], exist_ok=True)
    seed_everything(0)
    tr2 = IRPTrainer(cfg_b, device=cfg['training']['device'])
    tr2.load_checkpoint(ck)
    assert tr2.global_step == split_at
    logs2 = list(logs1)
    while tr2.global_step < int(cfg['training']['max_steps']):
        logs2.append(tr2.train_update())
    return tr2, logs2


def _compare(tr_a, logs_a, tr_b, logs_b, *, atol_param, atol_loss):
    assert tr_a.global_step == tr_b.global_step
    assert len(logs_a) == len(logs_b)
    for i, (la, lb) in enumerate(zip(logs_a, logs_b)):
        assert la['sample_ids'] == lb['sample_ids'], 'step %d sample_ids' % i
        assert la.get('aug_meta') == lb.get('aug_meta'), 'step %d aug' % i
        assert abs(la['learning_rate'] - lb['learning_rate']) < 1e-12
        assert abs(la['loss'] - lb['loss']) < atol_loss, (i, la['loss'], lb['loss'])
    max_param = 0.0
    for (n, p1), (_, p2) in zip(tr_a.model.named_parameters(), tr_b.model.named_parameters()):
        d = float((p1.detach().cpu() - p2.detach().cpu()).abs().max())
        max_param = max(max_param, d)
        if d > atol_param:
            raise AssertionError('param %s diff %s > %s' % (n, d, atol_param))
    # Optimizer / scheduler
    for k in tr_a.opt.state_dict():
        va, vb = tr_a.opt.state_dict()[k], tr_b.opt.state_dict()[k]
        if torch.is_tensor(va):
            assert torch.allclose(va.cpu(), vb.cpu(), atol=atol_param, rtol=1e-5), k
    assert tr_a.sched.state_dict()['updates_done'] == tr_b.sched.state_dict()['updates_done']
    return dict(max_param_diff=max_param)


def test_exact_resume_accum1_cpu():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg = _cfg(os.path.join(td, 'u'), synth, accum=1)
        tr_a, logs_a = _run_uninterrupted(copy.deepcopy(cfg))
        tr_b, logs_b = _run_interrupted(copy.deepcopy(cfg), split_at=3)
        stats = _compare(tr_a, logs_a, tr_b, logs_b,
                         atol_param=CPU_PARAM_ATOL, atol_loss=CPU_LOSS_ATOL)
        print('accum1', stats)


def test_exact_resume_accum2_cpu():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg = _cfg(os.path.join(td, 'u2'), synth, accum=2, max_steps=4)
        tr_a, logs_a = _run_uninterrupted(copy.deepcopy(cfg))
        tr_b, logs_b = _run_interrupted(copy.deepcopy(cfg), split_at=2)
        stats = _compare(tr_a, logs_a, tr_b, logs_b,
                         atol_param=CPU_PARAM_ATOL, atol_loss=CPU_LOSS_ATOL)
        print('accum2', stats)


def test_exact_resume_workers_amp_cuda():
    if not torch.cuda.is_available():
        try:
            import pytest
            pytest.skip('CUDA not available')
        except ImportError:
            raise SystemExit('SKIPPED: CUDA not available')
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg = _cfg(os.path.join(td, 'cu'), synth, accum=1, amp=True, num_workers=2,
                   max_steps=4, device='cuda')
        tr_a, logs_a = _run_uninterrupted(copy.deepcopy(cfg))
        tr_b, logs_b = _run_interrupted(copy.deepcopy(cfg), split_at=2)
        for la, lb in zip(logs_a, logs_b):
            assert la['sample_ids'] == lb['sample_ids']
            assert la.get('aug_meta') == lb.get('aug_meta')
            assert abs(la['learning_rate'] - lb['learning_rate']) < 1e-12
        max_param = 0.0
        for p1, p2 in zip(tr_a.model.parameters(), tr_b.model.parameters()):
            max_param = max(max_param, float((p1.cpu() - p2.cpu()).abs().max()))
        print('cuda amp/workers max_param_diff', max_param)
        if max_param > CUDA_AMP_PARAM_ATOL:
            raise AssertionError(
                'CUDA AMP resume param diff %s exceeds threshold %s'
                % (max_param, CUDA_AMP_PARAM_ATOL))


def main():
    test_exact_resume_accum1_cpu()
    test_exact_resume_accum2_cpu()
    try:
        test_exact_resume_workers_amp_cuda()
    except SystemExit as e:
        print(e)
    print('ALL test_irp_exact_resume PASS')


if __name__ == '__main__':
    main()
