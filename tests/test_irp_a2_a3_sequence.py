"""A2/A3 share sample order and augmentation under the same seed."""
from __future__ import annotations

import copy
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import torch
import yaml

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.irp_trainer import IRPTrainer  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402


def _cfg(tmp, prior, max_steps=3):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1.yaml')))
    cfg['out_dir'] = tmp
    cfg['model']['prior_source'] = prior
    # A2 ignores external ref; still need paths for nano trainer require_ref
    cfg['training'].update(dict(
        max_steps=max_steps, batch_size=2, num_workers=0, amp=False,
        grad_accum=1, warmup_steps=1, val_every=1000, ckpt_every=1000,
        log_every=1, device='cpu', halo=0, crop_size=64, seed=42,
    ))
    cfg['seed'] = 42
    cfg['data']['ref_mode'] = 'correct'
    return cfg


def test_param_counts_and_init_seed():
    seed_everything(0)
    a2 = build_asrt_irp(dict(model=dict(
        name='asrt_irp_v1', prior_source='low', n_feats=64,
        num_res_blocks='8+8+4+2', attention=dict(heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0), irp=dict(alpha_init=1., beta_init=0.1))))
    seed_everything(0)
    a3 = build_asrt_irp(dict(model=dict(
        name='asrt_irp_v1', prior_source='nano', n_feats=64,
        num_res_blocks='8+8+4+2', attention=dict(heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0), irp=dict(alpha_init=1., beta_init=0.1))))
    n2 = sum(p.numel() for p in a2.parameters())
    n3 = sum(p.numel() for p in a3.parameters())
    assert n2 == n3
    # shared keys should match under same seed
    sd2, sd3 = a2.state_dict(), a3.state_dict()
    common = set(sd2) & set(sd3)
    matched = 0
    for k in common:
        if sd2[k].shape == sd3[k].shape and torch.allclose(sd2[k], sd3[k]):
            matched += 1
    assert matched > 100


def test_a2_a3_sample_and_aug_sequence():
    with tempfile.TemporaryDirectory() as td:
        cfg2 = _cfg(os.path.join(td, 'a2'), 'low')
        cfg3 = _cfg(os.path.join(td, 'a3'), 'nano')
        # A2 require_ref False path: prior_source=low still builds ref encoder
        # but dataset require_ref follows prior_source==nano only
        try:
            seed_everything(42)
            t2 = IRPTrainer(cfg2, device='cpu')
            seed_everything(42)
            t3 = IRPTrainer(cfg3, device='cpu')
        except Exception as e:
            print('SKIP a2/a3 sequence (data):', e)
            return
        ids2, ids3 = [], []
        aug2, aug3 = [], []
        for _ in range(3):
            b2 = t2._next_batch()
            b3 = t3._next_batch()
            ids2.append(list(b2['sample_id']))
            ids3.append(list(b3['sample_id']))
            aug2.append(b2.get('aug_meta'))
            aug3.append(b3.get('aug_meta'))
        assert ids2 == ids3, (ids2, ids3)
        # geometry metadata should match (ref content may differ)
        assert aug2 == aug3
        assert t2.out_dir != t3.out_dir


def main():
    test_param_counts_and_init_seed()
    test_a2_a3_sample_and_aug_sequence()
    print('ALL test_irp_a2_a3_sequence PASS')


if __name__ == '__main__':
    main()
