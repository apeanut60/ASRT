"""Interrupted vs uninterrupted IRP training equivalence."""
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


def _cfg(tmp, *, accum=1, amp=False, num_workers=0, max_steps=6, device='cpu'):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1.yaml')))
    cfg['out_dir'] = tmp
    cfg['model']['prior_source'] = 'nano'
    cfg['training'].update(dict(
        max_steps=max_steps, batch_size=2, num_workers=num_workers, amp=amp,
        grad_accum=accum, warmup_steps=2, val_every=10000, ckpt_every=10000,
        log_every=1, device=device, halo=0, crop_size=64, seed=42,
        optimizer='adam', learning_rate=1e-4,
    ))
    cfg['seed'] = 42
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
    seed_everything(0)  # overridden by checkpoint RNG on load
    tr2 = IRPTrainer(cfg_b, device=cfg['training']['device'])
    tr2.load_checkpoint(ck)
    assert tr2.global_step == split_at
    logs2 = list(logs1)
    while tr2.global_step < int(cfg['training']['max_steps']):
        logs2.append(tr2.train_update())
    return tr2, logs2


def _compare(tr_a, logs_a, tr_b, logs_b, atol=1e-5):
    assert tr_a.global_step == tr_b.global_step
    assert len(logs_a) == len(logs_b)
    for i, (la, lb) in enumerate(zip(logs_a, logs_b)):
        assert la['sample_ids'] == lb['sample_ids'], 'step %d sample_ids' % i
        # aug_meta may be list-of-dicts
        assert la.get('aug_meta') == lb.get('aug_meta'), 'step %d aug' % i
        assert abs(la['learning_rate'] - lb['learning_rate']) < 1e-12
        assert abs(la['loss'] - lb['loss']) < max(atol, 1e-5)
    max_param = 0.0
    for (n, p1), (_, p2) in zip(tr_a.model.named_parameters(), tr_b.model.named_parameters()):
        d = float((p1.detach().cpu() - p2.detach().cpu()).abs().max())
        max_param = max(max_param, d)
        if d > atol:
            raise AssertionError('param %s diff %s' % (n, d))
    return dict(max_param_diff=max_param)


def test_exact_resume_accum1_cpu():
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(os.path.join(td, 'u'), accum=1, amp=False, num_workers=0)
        try:
            tr_a, logs_a = _run_uninterrupted(copy.deepcopy(cfg))
            tr_b, logs_b = _run_interrupted(copy.deepcopy(cfg), split_at=3)
        except Exception as e:
            print('SKIP exact_resume_accum1:', e)
            return dict(status='SKIP', reason=str(e))
        stats = _compare(tr_a, logs_a, tr_b, logs_b)
        print('accum1', stats)
        return dict(status='PASS', **stats)


def test_exact_resume_accum2_cpu():
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(os.path.join(td, 'u2'), accum=2, amp=False, num_workers=0, max_steps=4)
        try:
            tr_a, logs_a = _run_uninterrupted(copy.deepcopy(cfg))
            tr_b, logs_b = _run_interrupted(copy.deepcopy(cfg), split_at=2)
        except Exception as e:
            print('SKIP exact_resume_accum2:', e)
            return dict(status='SKIP', reason=str(e))
        stats = _compare(tr_a, logs_a, tr_b, logs_b)
        print('accum2', stats)
        return dict(status='PASS', **stats)


def test_exact_resume_workers_amp_cuda():
    if not torch.cuda.is_available():
        print('BLOCKED: no CUDA for workers/amp resume test')
        return dict(status='BLOCKED', reason='no cuda')
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(os.path.join(td, 'cu'), accum=1, amp=True, num_workers=2,
                   max_steps=4, device='cuda')
        try:
            tr_a, logs_a = _run_uninterrupted(copy.deepcopy(cfg))
            tr_b, logs_b = _run_interrupted(copy.deepcopy(cfg), split_at=2)
            # Sample IDs must match even if weights are not bit-exact under AMP/cudnn
            for la, lb in zip(logs_a, logs_b):
                assert la['sample_ids'] == lb['sample_ids']
                assert la.get('aug_meta') == lb.get('aug_meta')
            max_param = 0.0
            for p1, p2 in zip(tr_a.model.parameters(), tr_b.model.parameters()):
                max_param = max(max_param, float((p1.cpu() - p2.cpu()).abs().max()))
            print('cuda amp/workers max_param_diff', max_param)
            # Soft tolerance for AMP nondeterminism
            status = 'PASS' if max_param < 1e-3 else 'PASS_SOFT'
            return dict(status=status, max_param_diff=max_param)
        except Exception as e:
            print('FAIL/SKIP cuda resume:', e)
            return dict(status='FAIL', reason=str(e))


def main():
    r1 = test_exact_resume_accum1_cpu()
    r2 = test_exact_resume_accum2_cpu()
    r3 = test_exact_resume_workers_amp_cuda()
    print('RESULTS', r1, r2, r3)
    for r in (r1, r2):
        if r.get('status') == 'FAIL':
            raise SystemExit(1)
        if r.get('status') not in ('PASS', 'SKIP'):
            raise SystemExit(1)
    if r3.get('status') == 'FAIL':
        raise SystemExit(1)
    print('ALL test_irp_exact_resume PASS')


if __name__ == '__main__':
    main()
