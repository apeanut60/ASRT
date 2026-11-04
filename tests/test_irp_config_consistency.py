"""Resume config / ref_mode / validation protocol consistency (fail-fast)."""
from __future__ import annotations

import copy
import os
import sys
import tempfile

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import extract_irp_hparams  # noqa: E402
from asrt.models.factory import compare_irp_hparams  # noqa: E402
from asrt.training.irp_trainer import (  # noqa: E402
    IRPTrainer,
    compare_train_protocol,
    extract_train_protocol,
    extract_validation_protocol,
)
from tests.irp_test_fixtures import apply_synth_paths, make_synthetic_triplet_dataset  # noqa: E402


def _base_cfg(tmp, synth, prior='nano', max_steps=3):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1.yaml')))
    cfg = apply_synth_paths(cfg, synth)
    cfg['out_dir'] = tmp
    cfg['model']['prior_source'] = prior
    cfg['data']['ref_mode'] = 'correct'
    cfg['training'].update(dict(
        max_steps=max_steps, batch_size=2, num_workers=0, amp=False,
        grad_accum=1, warmup_steps=1, val_every=1000, ckpt_every=1000,
        log_every=1, device='cpu', halo=0, crop_size=64, seed=42,
        drop_last=True, val_max_images=2,
    ))
    cfg['evaluation'] = dict(
        inference_mode='full', irp_allow_tiled=False, irp_oom_policy='error',
        input_range='minus_one_one', tile=256, overlap=32, full_max_side=100000,
    )
    return cfg


def test_compare_helpers():
    a = extract_irp_hparams(dict(model=dict(
        name='asrt_irp_v1', prior_source='nano', n_feats=64,
        num_res_blocks='8+8+4+2', enable_lightup=True,
        attention=dict(type='window_spatial', heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    )))
    b = dict(a); b['prior_source'] = 'low'
    assert compare_irp_hparams(a, b)
    cfg = dict(
        training=dict(max_steps=10, learning_rate=1e-4, batch_size=2,
                      grad_accum=2, seed=42, val_every=100),
        model=dict(prior_source='nano'),
        data=dict(ref_mode='correct'),
        evaluation=dict(inference_mode='full', irp_allow_tiled=False),
    )
    p = extract_train_protocol(cfg, train_hash='x', val_hash='y', dataset_size=10, halo=32)
    assert p['ref_mode'] == 'correct'
    assert 'validation_protocol_hash' in p
    p2 = dict(p); p2['grad_accum'] = 1
    assert compare_train_protocol(p, p2)


def _assert_resume_fails(ck, cfg_mut):
    tr2 = IRPTrainer(cfg_mut, device='cpu')
    # Snapshot a marker weight before load attempt
    w0 = next(tr2.model.parameters()).detach().clone()
    raised = False
    try:
        tr2.load_checkpoint(ck)
    except RuntimeError as e:
        raised = True
        assert 'mismatch' in str(e).lower() or 'protocol' in str(e).lower()
    assert raised, 'expected resume to fail'
    # Trainer weights must be unchanged after failed resume
    assert torch_equalish(next(tr2.model.parameters()), w0)


def torch_equalish(a, b):
    import torch
    return torch.equal(a.cpu(), b.cpu())


def test_resume_protocol_bindings():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg = _base_cfg(os.path.join(td, 'run'), synth)
        tr = IRPTrainer(cfg, device='cpu')
        tr.fit()
        ck = tr.save_checkpoint('last')

        # Same config → PASS
        cfg_ok = copy.deepcopy(cfg)
        cfg_ok['out_dir'] = os.path.join(td, 'ok')
        tr_ok = IRPTrainer(cfg_ok, device='cpu')
        tr_ok.load_checkpoint(ck)
        assert tr_ok.global_step == tr.global_step

        cases = []
        # Correct → Ref=Low
        c = copy.deepcopy(cfg); c['out_dir'] = os.path.join(td, 'ref_low')
        c['data']['ref_mode'] = 'ref_low'
        c['training']['allow_non_correct_ref'] = True
        cases.append(('ref_low', c))
        # Correct → Wrong
        c = copy.deepcopy(cfg); c['out_dir'] = os.path.join(td, 'wrong')
        c['data']['ref_mode'] = 'wrong'
        c['training']['allow_non_correct_ref'] = True
        cases.append(('wrong', c))
        # Full → Tiled
        c = copy.deepcopy(cfg); c['out_dir'] = os.path.join(td, 'tiled')
        c['evaluation']['inference_mode'] = 'tiled'
        c['evaluation']['irp_allow_tiled'] = True
        cases.append(('tiled', c))
        # input_range
        c = copy.deepcopy(cfg); c['out_dir'] = os.path.join(td, 'range')
        c['evaluation']['input_range'] = 'zero_one'
        cases.append(('input_range', c))
        # val_max_images
        c = copy.deepcopy(cfg); c['out_dir'] = os.path.join(td, 'vmax')
        c['training']['val_max_images'] = 1
        cases.append(('val_max_images', c))
        # val_every
        c = copy.deepcopy(cfg); c['out_dir'] = os.path.join(td, 'vevery')
        c['training']['val_every'] = 50
        cases.append(('val_every', c))
        # LR change
        c = copy.deepcopy(cfg); c['out_dir'] = os.path.join(td, 'lr')
        c['training']['learning_rate'] = 2e-4
        cases.append(('lr', c))

        for name, cmut in cases:
            print('expect fail:', name)
            _assert_resume_fails(ck, cmut)

        # Manifest change
        synth2 = make_synthetic_triplet_dataset(
            os.path.join(td, 'data2'), n_train=8, n_val=4, size=160, seed=99)
        c = copy.deepcopy(cfg)
        c = apply_synth_paths(c, synth2)
        c['out_dir'] = os.path.join(td, 'man')
        print('expect fail: manifest')
        _assert_resume_fails(ck, c)


def test_formal_rejects_non_correct_ref_mode():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg = _base_cfg(os.path.join(td, 'bad'), synth)
        cfg['data']['ref_mode'] = 'wrong'
        raised = False
        try:
            IRPTrainer(cfg, device='cpu')
        except ValueError as e:
            raised = True
            assert 'ref_mode' in str(e)
        assert raised


def main():
    test_compare_helpers()
    test_resume_protocol_bindings()
    test_formal_rejects_non_correct_ref_mode()
    print('ALL test_irp_config_consistency PASS')


if __name__ == '__main__':
    main()
