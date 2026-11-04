"""Fresh-start out_dir overwrite protection."""
from __future__ import annotations

import os
import sys
import tempfile

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.training.irp_trainer import IRPTrainer  # noqa: E402
from tests.irp_test_fixtures import apply_synth_paths, make_synthetic_triplet_dataset  # noqa: E402


def _cfg(tmp, synth):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1.yaml')))
    cfg = apply_synth_paths(cfg, synth)
    cfg['out_dir'] = tmp
    cfg['model']['prior_source'] = 'low'
    cfg['data']['ref_mode'] = 'correct'
    cfg['training'].update(dict(
        max_steps=2, batch_size=2, num_workers=0, amp=False,
        grad_accum=1, warmup_steps=1, val_every=1000, ckpt_every=1000,
        log_every=1, device='cpu', halo=0, crop_size=64, seed=0,
        drop_last=True,
    ))
    return cfg


def test_fresh_start_refuses_existing_artifacts():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg = _cfg(os.path.join(td, 'exp'), synth)
        tr = IRPTrainer(cfg, device='cpu')
        tr.train_update()
        tr.save_checkpoint('x')
        raised = False
        try:
            IRPTrainer(cfg, device='cpu')
        except RuntimeError as e:
            raised = True
            assert 'Refuse fresh start' in str(e)
        assert raised


def test_resume_flag_allows_existing():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg = _cfg(os.path.join(td, 'exp2'), synth)
        tr = IRPTrainer(cfg, device='cpu')
        tr.train_update()
        ck = tr.save_checkpoint('x')
        cfg2 = dict(cfg)
        cfg2['_allow_existing_out_dir'] = True
        tr2 = IRPTrainer(cfg2, device='cpu')
        tr2.load_checkpoint(ck)
        assert tr2.global_step == 1
