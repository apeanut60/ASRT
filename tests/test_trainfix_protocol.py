"""Seed / eval-protocol / hparam consistency tests for trainfix."""
from __future__ import annotations

import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.evaluation.protocol import infer_with_protocol, resolve_eval_cfg
from asrt.models.asrt_v4 import build_asrt_v4
from asrt.models.factory import build_and_load, compare_v4_hparams, _extract_v4_hparams
from asrt.training.seed_utils import seed_everything
from asrt.training.scheduler import preview_lrs
from asrt.training.optimizer import build_optimizer


def test_seed_reproducible_init():
    seed_everything(123)
    a = torch.rand(4)
    seed_everything(123)
    b = torch.rand(4)
    assert torch.equal(a, b)
    seed_everything(124)
    c = torch.rand(4)
    assert not torch.equal(a, c)


def test_eval_protocol_modes():
    cfg = dict(evaluation=dict(inference_mode='auto', full_max_side=100, tile=64, overlap=8))
    ev = resolve_eval_cfg(cfg)
    assert ev['inference_mode'] == 'auto'
    m = build_asrt_v4().eval()
    small = torch.rand(1, 3, 64, 64) * 2 - 1
    big = torch.rand(1, 3, 128, 128) * 2 - 1
    with torch.no_grad():
        _, meta_s = infer_with_protocol(m, small, None, eval_cfg=cfg)
        _, meta_b = infer_with_protocol(m, big, None, eval_cfg=cfg)
    assert meta_s['inference_used'] == 'full'
    assert meta_b['inference_used'] == 'tiled'


def test_hparam_mismatch_rejects():
    m = build_asrt_v4()
    path = tempfile.mktemp(suffix='.pt')
    cfg = dict(stage='b_v4', model=dict(
        version='v4', enable_v4=True, n_feats=64, num_res_blocks='8+8+4+2',
        enable_lightup=True, match_dim=64, match_radius=4,
        match_temperature=0.1, match_lambda_pos=0.5, ref_base_ch=32, reli_mid=64,
        matching=dict(dim=64, radius=4, temperature=0.1, lambda_pos=0.5),
    ))
    torch.save(dict(model=m.state_dict(), step=1, cfg=cfg), path)
    bad = dict(cfg)
    bad['model'] = dict(cfg['model'])
    bad['model']['matching'] = dict(cfg['model']['matching'])
    bad['model']['matching']['radius'] = 8
    bad['model']['match_radius'] = 8
    try:
        build_and_load(bad, path, device='cpu')
        assert False, 'should reject'
    except RuntimeError as e:
        msg = str(e).lower()
        assert 'match_radius' in msg or 'hyperparam' in msg or 'conflict' in msg, msg


def test_trainer_evaluate_matches_protocol():
    """Trainer.evaluate and infer_with_protocol agree on one synthetic image."""
    from asrt.training.trainer import Trainer
    from asrt.evaluation.metrics import psnr_ssim_rgb
    import numpy as np
    from PIL import Image
    import asrt.training.trainer as Tmod

    tmp = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    low_p = os.path.join(tmp, 'low.png')
    gt_p = os.path.join(tmp, 'gt.png')
    arr = np.full((64, 64, 3), 40, np.uint8)
    Image.fromarray(arr).save(low_p)
    Image.fromarray(np.full((64, 64, 3), 80, np.uint8)).save(gt_p)
    rows = [dict(sample_id='x', low_path=low_p, gt_path=gt_p, domain='real')]

    real_build = Tmod.build_asrt
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)
            self.is_v4 = False
            self.enable_v3 = False
            self.backbone = torch.nn.Identity()
            self.last_aux = None
        def forward(self, low, ref=None, **kw):
            return torch.tanh(self.conv(low))
    Tmod.build_asrt = lambda cfg=None: Tiny()
    try:
        cfg = dict(
            stage='a', seed=0, max_steps=1, batch_size=1, accum=1,
            amp=False, workers=0, out_dir=out, crop=64,
            evaluation=dict(inference_mode='full'),
            model=dict(),
        )
        tr = Trainer(cfg, device='cpu')
        m1 = tr.evaluate(rows, with_ref=False)
        from asrt.data.paired import read_rgb_m11
        low = read_rgb_m11(low_p)[None]
        gt = read_rgb_m11(gt_p)[None]
        pred, _ = infer_with_protocol(tr.model, low, None, eval_cfg=cfg)
        m2 = psnr_ssim_rgb(pred, gt)
        assert abs(m1['psnr_rgb'] - m2['psnr_rgb']) < 1e-6
        assert m1['n'] == 1
    finally:
        Tmod.build_asrt = real_build


def test_lr_preview_v4_groups():
    m = build_asrt_v4()
    opt = build_optimizer(m, dict(
        stage='b_v4', lr=1e-5, lr_backbone=1e-5, lr_v4=1e-4,
        lr_igab=2e-5, lr_illum=2e-5, lr_inject=1e-5, eta_min=1e-6,
        total_steps=100000))
    prev = preview_lrs(opt, dict(total_steps=100000, eta_min=1e-6, lr=1e-5))
    end = list(prev.values())[-1]
    for name, lr in end.items():
        assert abs(lr - 1e-6) < 1e-9, (name, lr)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print('running', name, '...')
            fn()
            print('  ok')
    print('ALL PASS')
