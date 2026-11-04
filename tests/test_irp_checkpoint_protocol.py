"""IRP checkpoint identity / prior_source consistency."""
from __future__ import annotations

import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp, extract_irp_hparams  # noqa: E402
from asrt.models.factory import (  # noqa: E402
    build_and_load,
    detect_ckpt_version,
    load_checkpoint_strict,
)


def _cfg(src):
    return dict(
        stage='irp_v1',
        model=dict(
            name='asrt_irp_v1', version='irp_v1', model_family='asrt_irp',
            n_feats=64, num_res_blocks='8+8+4+2', prior_source=src,
            enable_lightup=True,
            attention=dict(type='window_spatial', heads=4, window_size=8),
            reference_prior=dict(gaussian_sigma=7.0),
            irp=dict(alpha_init=1.0, beta_init=0.1),
        ))


def _save(model, cfg, path):
    torch.save(dict(
        model=model.state_dict(),
        cfg=cfg,
        model_family='asrt_irp',
        model_version='irp_v1',
        prior_source=cfg['model']['prior_source'],
        irp_hparams=extract_irp_hparams(cfg),
        step=0,
    ), path)


def test_a3_to_a3_pass():
    cfg = _cfg('nano')
    m = build_asrt_irp(cfg)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'a3.pt')
        _save(m, cfg, path)
        m2 = build_asrt_irp(cfg)
        rep = load_checkpoint_strict(m2, path, external_cfg=cfg)
        assert rep['version'] == 'irp_v1'


def test_a3_to_a2_fail():
    cfg3 = _cfg('nano')
    cfg2 = _cfg('low')
    m = build_asrt_irp(cfg3)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'a3.pt')
        _save(m, cfg3, path)
        m2 = build_asrt_irp(cfg2)
        try:
            load_checkpoint_strict(m2, path, external_cfg=cfg2)
            raise AssertionError('expected mismatch')
        except RuntimeError as e:
            assert 'prior_source' in str(e) or 'hyperparam' in str(e)


def test_window_and_sigma_mismatch():
    cfg = _cfg('nano')
    m = build_asrt_irp(cfg)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'a3.pt')
        _save(m, cfg, path)
        bad = _cfg('nano')
        bad['model']['attention']['window_size'] = 16
        m2 = build_asrt_irp(bad)
        try:
            load_checkpoint_strict(m2, path, external_cfg=bad)
            raise AssertionError('expected window mismatch')
        except RuntimeError:
            pass
        bad2 = _cfg('nano')
        bad2['model']['reference_prior']['gaussian_sigma'] = 3.0
        # sigma is not a weight shape — still must fail hparam check
        m3 = build_asrt_irp(bad2)
        try:
            load_checkpoint_strict(m3, path, external_cfg=bad2)
            raise AssertionError('expected sigma mismatch')
        except RuntimeError:
            pass


def test_missing_hparams_fail():
    cfg = _cfg('low')
    m = build_asrt_irp(cfg)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'bare.pt')
        torch.save(dict(model=m.state_dict()), path)
        m2 = build_asrt_irp(cfg)
        try:
            load_checkpoint_strict(m2, path)
            raise AssertionError('expected missing hparams failure')
        except RuntimeError as e:
            assert 'missing' in str(e).lower() or 'irp' in str(e).lower()


def test_detect_from_state_keys():
    m = build_asrt_irp(_cfg('nano'))
    state = m.state_dict()
    assert detect_ckpt_version(state, None) == 'irp_v1'


def test_a2_to_a2_pass():
    cfg = _cfg('low')
    m = build_asrt_irp(cfg)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'a2.pt')
        _save(m, cfg, path)
        _, rep = build_and_load(cfg, path)
        assert rep['version'] == 'irp_v1'


def main():
    test_a3_to_a3_pass()
    test_a3_to_a2_fail()
    test_window_and_sigma_mismatch()
    test_missing_hparams_fail()
    test_detect_from_state_keys()
    test_a2_to_a2_pass()
    print('ALL test_irp_checkpoint_protocol PASS')


if __name__ == '__main__':
    main()
