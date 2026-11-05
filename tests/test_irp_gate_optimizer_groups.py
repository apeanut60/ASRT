"""Gate Adam parameter-group partitioning."""
from __future__ import annotations

import copy
import os
import sys
import tempfile

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.training.irp_trainer import IRPTrainer, split_irp_gate_params  # noqa: E402
from tests.irp_test_fixtures import apply_synth_paths, make_synthetic_triplet_dataset  # noqa: E402


def _base(tmp, synth, **gate_kw):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1.yaml')))
    cfg = apply_synth_paths(cfg, synth)
    cfg['out_dir'] = tmp
    cfg['model']['prior_source'] = 'nano'
    cfg['data']['ref_mode'] = 'correct'
    cfg['training'].update(dict(
        max_steps=2, batch_size=2, num_workers=0, amp=False,
        grad_accum=1, warmup_steps=1, val_every=1000, ckpt_every=1000,
        log_every=1, device='cpu', halo=0, crop_size=64, seed=0,
        drop_last=True, learning_rate=1e-4, weight_decay=1e-4,
    ))
    cfg['training'].update(gate_kw)
    return cfg


def test_split_no_overlap_no_missing():
    from asrt.models.asrt_irp import build_asrt_irp
    m = build_asrt_irp(dict(model=dict(
        name='asrt_irp_v1', prior_source='nano', n_feats=64,
        num_res_blocks='8+8+4+2', attention=dict(heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0),
    )))
    gates, others, names = split_irp_gate_params(m)
    ids_g = {id(p) for p in gates}
    ids_o = {id(p) for p in others}
    assert ids_g.isdisjoint(ids_o)
    all_train = {id(p) for p in m.parameters() if p.requires_grad}
    assert ids_g | ids_o == all_train
    assert len(names) == 5  # 3 alpha + 2 beta
    assert any(n.endswith('.alpha') for n in names)
    assert any(n.endswith('.beta') for n in names)


def test_b0_single_group_equivalent():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg = _base(os.path.join(td, 'b0'), synth)  # no gate_* overrides
        tr = IRPTrainer(cfg, device='cpu')
        assert tr.opt_param_groups_mode == 'single_v1_equivalent'
        assert len(tr.opt.param_groups) == 1
        assert abs(tr.opt.param_groups[0]['lr'] - 1e-4) < 1e-15
        assert abs(tr.opt.param_groups[0]['weight_decay'] - 1e-4) < 1e-15


def test_b1_b2_only_change_gate_hparams():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(os.path.join(td, 'data'), size=160)
        cfg1 = _base(os.path.join(td, 'b1'), synth, gate_lr=1e-4, gate_weight_decay=0.0)
        tr1 = IRPTrainer(cfg1, device='cpu')
        assert tr1.opt_param_groups_mode == 'split_gates'
        assert len(tr1.opt.param_groups) == 2
        # group0 backbone, group1 gates
        assert abs(tr1.opt.param_groups[0]['weight_decay'] - 1e-4) < 1e-15
        assert abs(tr1.opt.param_groups[1]['weight_decay'] - 0.0) < 1e-15
        assert abs(tr1.opt.param_groups[1]['lr'] - 1e-4) < 1e-15

        cfg2 = _base(os.path.join(td, 'b2'), synth, gate_lr=1e-5, gate_weight_decay=0.0)
        tr2 = IRPTrainer(cfg2, device='cpu')
        assert abs(tr2.opt.param_groups[0]['lr'] - 1e-4) < 1e-15
        assert abs(tr2.opt.param_groups[1]['lr'] - 1e-5) < 1e-15
        assert abs(tr2.opt.param_groups[1]['weight_decay'] - 0.0) < 1e-15
