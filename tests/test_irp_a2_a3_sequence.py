"""A2/A3 identical init weights + shared sample/aug/LR sequence (fail-fast)."""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.irp_trainer import IRPTrainer  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402
from tests.irp_test_fixtures import apply_synth_paths, make_synthetic_triplet_dataset  # noqa: E402


def _model_cfg(prior):
    return dict(model=dict(
        name='asrt_irp_v1', prior_source=prior, n_feats=64,
        num_res_blocks='8+8+4+2', attention=dict(heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    ))


def _trainer_cfg(tmp, synth, prior, max_steps=3):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1.yaml')))
    cfg = apply_synth_paths(cfg, synth)
    cfg['out_dir'] = tmp
    cfg['model']['prior_source'] = prior
    cfg['data']['ref_mode'] = 'correct'
    cfg['training'].update(dict(
        max_steps=max_steps, batch_size=2, num_workers=0, amp=False,
        grad_accum=1, warmup_steps=1, val_every=1000, ckpt_every=1000,
        log_every=1, device='cpu', halo=0, crop_size=64, seed=42,
        drop_last=True,
    ))
    cfg['seed'] = 42
    return cfg


def test_a2_a3_all_weights_equal_under_same_seed():
    seed_everything(0)
    a2 = build_asrt_irp(_model_cfg('low'))
    seed_everything(0)
    a3 = build_asrt_irp(_model_cfg('nano'))
    sd2, sd3 = a2.state_dict(), a3.state_dict()
    assert set(sd2.keys()) == set(sd3.keys())
    for k in sd2:
        assert sd2[k].shape == sd3[k].shape, k
        assert torch.equal(sd2[k], sd3[k]), 'mismatch at %s' % k
    assert sum(p.numel() for p in a2.parameters()) == sum(
        p.numel() for p in a3.parameters())


def test_a2_a3_sample_aug_lr_budget():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg2 = _trainer_cfg(os.path.join(td, 'a2'), synth, 'low', max_steps=3)
        cfg3 = _trainer_cfg(os.path.join(td, 'a3'), synth, 'nano', max_steps=3)
        seed_everything(42)
        t2 = IRPTrainer(cfg2, device='cpu')
        seed_everything(42)
        t3 = IRPTrainer(cfg3, device='cpu')
        assert t2.max_steps == t3.max_steps
        assert t2.accum == t3.accum
        assert t2.batch_size == t3.batch_size
        assert t2.out_dir != t3.out_dir

        ids2, ids3, aug2, aug3, lr2, lr3 = [], [], [], [], [], []
        for _ in range(3):
            log2 = t2.train_update()
            log3 = t3.train_update()
            ids2.append(log2['sample_ids'])
            ids3.append(log3['sample_ids'])
            aug2.append(log2.get('aug_meta'))
            aug3.append(log3.get('aug_meta'))
            lr2.append(log2['learning_rate'])
            lr3.append(log3['learning_rate'])
        assert ids2 == ids3, (ids2, ids3)
        assert aug2 == aug3
        assert lr2 == lr3
        assert t2.global_step == t3.global_step == 3


def main():
    test_a2_a3_all_weights_equal_under_same_seed()
    test_a2_a3_sample_aug_lr_budget()
    print('ALL test_irp_a2_a3_sequence PASS')


if __name__ == '__main__':
    main()
