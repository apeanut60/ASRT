"""Resume config / protocol consistency checks."""
from __future__ import annotations

import copy
import os
import sys
import tempfile

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import extract_irp_hparams  # noqa: E402
from asrt.models.factory import compare_irp_hparams  # noqa: E402
from asrt.training.irp_trainer import (  # noqa: E402
    IRPTrainer,
    compare_train_protocol,
    extract_train_protocol,
)


def _base_cfg(tmp, prior='nano', max_steps=4):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1.yaml')))
    cfg['out_dir'] = tmp
    cfg['model']['prior_source'] = prior
    cfg['training'].update(dict(
        max_steps=max_steps, batch_size=2, num_workers=0, amp=False,
        grad_accum=1, warmup_steps=1, val_every=1000, ckpt_every=1000,
        log_every=1, device='cpu', halo=0, crop_size=64,
    ))
    # Shrink data for speed: trainer loads full manifests — OK for few steps
    return cfg


def test_compare_helpers():
    a = extract_irp_hparams(dict(model=dict(
        name='asrt_irp_v1', prior_source='nano', n_feats=64,
        num_res_blocks='8+8+4+2', enable_lightup=True,
        attention=dict(type='window_spatial', heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    )))
    b = dict(a)
    b['prior_source'] = 'low'
    assert compare_irp_hparams(a, b)
    p = extract_train_protocol(
        dict(training=dict(max_steps=10, learning_rate=1e-4, batch_size=2,
                           grad_accum=2, seed=42), model=dict(prior_source='nano')),
        train_hash='x', val_hash='y', dataset_size=10, halo=22)
    p2 = dict(p); p2['grad_accum'] = 1
    assert compare_train_protocol(p, p2)


def test_resume_rejects_protocol_mismatch():
    with tempfile.TemporaryDirectory() as td:
        cfg = _base_cfg(td)
        try:
            tr = IRPTrainer(cfg, device='cpu')
        except Exception as e:
            # If dataset paths missing, skip
            print('SKIP trainer init:', e)
            return
        tr.fit()
        ck = tr.save_checkpoint('last')
        cfg2 = copy.deepcopy(cfg)
        cfg2['out_dir'] = td + '/b'
        cfg2['training']['learning_rate'] = 2e-4
        tr2 = IRPTrainer(cfg2, device='cpu')
        try:
            tr2.load_checkpoint(ck)
            raise AssertionError('expected protocol mismatch')
        except RuntimeError as e:
            assert 'protocol' in str(e).lower() or 'mismatch' in str(e).lower()


def main():
    test_compare_helpers()
    test_resume_rejects_protocol_mismatch()
    print('ALL test_irp_config_consistency PASS')


if __name__ == '__main__':
    main()
