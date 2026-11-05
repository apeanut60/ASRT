"""Eval loader must reject A2/A3 prior_source mismatches."""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp, extract_irp_hparams  # noqa: E402
from asrt.models.factory import load_checkpoint_strict  # noqa: E402


def _save_ckpt(path, prior):
    cfg = dict(model=dict(
        name='asrt_irp_v1', version='irp_v1', model_family='asrt_irp',
        prior_source=prior, n_feats=64, num_res_blocks='8+8+4+2',
        enable_lightup=True,
        attention=dict(type='window_spatial', heads=4, window_size=8),
        reference_prior=dict(gaussian_sigma=7.0),
        irp=dict(alpha_init=1.0, beta_init=0.1),
    ), training=dict(max_steps=1))
    m = build_asrt_irp(cfg)
    torch.save(dict(
        model=m.state_dict(),
        cfg=cfg,
        irp_hparams=extract_irp_hparams(cfg),
        model_family='asrt_irp',
        model_version='irp_v1',
        prior_source=prior,
        global_step=0,
    ), path)
    return cfg


def test_matching_prior_loads():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'a3.pt')
        cfg = _save_ckpt(path, 'nano')
        m = build_asrt_irp(cfg)
        rep = load_checkpoint_strict(
            m, path, expect_version='irp_v1', external_cfg=cfg)
        assert rep['version'] == 'irp_v1'


def test_a3_ckpt_rejected_by_a2_cfg():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'a3.pt')
        cfg_a3 = _save_ckpt(path, 'nano')
        cfg_a2 = yaml.safe_load(open(os.path.join(ROOT, 'configs/asrt_irp_v1_a2.yaml')))
        # slim model cfg for build
        cfg_a2_m = dict(model=dict(cfg_a2['model']))
        m = build_asrt_irp(cfg_a2_m)
        raised = False
        try:
            load_checkpoint_strict(
                m, path, expect_version='irp_v1', external_cfg=cfg_a2_m)
        except RuntimeError as e:
            raised = True
            assert 'prior_source' in str(e) or 'mismatch' in str(e).lower()
        assert raised


def test_a2_ckpt_rejected_by_a3_cfg():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'a2.pt')
        _save_ckpt(path, 'low')
        cfg_a3 = dict(model=dict(
            name='asrt_irp_v1', prior_source='nano', n_feats=64,
            num_res_blocks='8+8+4+2', enable_lightup=True,
            attention=dict(type='window_spatial', heads=4, window_size=8),
            reference_prior=dict(gaussian_sigma=7.0),
            irp=dict(alpha_init=1.0, beta_init=0.1),
        ))
        m = build_asrt_irp(cfg_a3)
        raised = False
        try:
            load_checkpoint_strict(
                m, path, expect_version='irp_v1', external_cfg=cfg_a3)
        except RuntimeError:
            raised = True
        assert raised
