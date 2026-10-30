"""ASRT-v4 functional / gradient / smoke tests."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_v4 import ASRTv4, build_asrt_v4, load_stage_a_into_v4
from asrt.models.csfi_unet import CSFIUNet, ResBlock
from asrt.models.v4_adapters import IlluminationAdapter, TextureAdapter
from asrt.models.v4_matching import LocalSoftMatcher, warp_by_disp
from asrt.training.optimizer import build_optimizer


def _rand_img(b=2, h=64, w=64):
    return (torch.rand(b, 3, h, w) * 2 - 1)


def test_local_soft_matcher_shapes_and_grad():
    m = LocalSoftMatcher(dim=64, radius=2, temperature=0.1)
    qf = torch.randn(2, 64, 16, 16, requires_grad=True)
    qi = torch.randn(2, 64, 16, 16, requires_grad=True)
    rf = torch.randn(2, 64, 16, 16, requires_grad=True)
    t, w, diag = m(qf, qi, rf)
    assert t.shape == (2, 64, 16, 16)
    assert w.shape == (2, 25, 16, 16)
    assert torch.allclose(w.sum(dim=1), torch.ones(2, 16, 16), atol=1e-4)
    loss = t.mean()
    loss.backward()
    assert qf.grad is not None and qf.grad.abs().sum() > 0
    assert rf.grad is not None and rf.grad.abs().sum() > 0
    assert m.query_proj[0].weight.grad is not None
    assert m.key_proj[0].weight.grad is not None


def test_boundary_mask_nonzero_weights():
    m = LocalSoftMatcher(dim=32, radius=3)
    qf = torch.randn(1, 32, 8, 8)
    qi = torch.randn(1, 32, 8, 8)
    rf = torch.randn(1, 32, 8, 8)
    _, w, _ = m(qf, qi, rf)
    # Corner must still sum to 1 over valid candidates
    assert torch.allclose(w.sum(dim=1), torch.ones(1, 8, 8), atol=1e-4)


def test_warp_nonsquare():
    feat = torch.randn(1, 16, 30, 40)
    disp = torch.zeros(1, 2, 15, 20)
    disp[:, 0] = 1.0  # shift +1 H/4 cell in x
    out = warp_by_disp(feat, disp, scale=2.0)
    assert out.shape == feat.shape


def test_zero_init_adapters():
    ta = TextureAdapter(32)
    ia = IlluminationAdapter(32)
    f = torch.randn(2, 32, 8, 8)
    t = torch.randn(2, 32, 8, 8)
    c = torch.ones(2, 1, 8, 8)
    out = ta(f, t, c)
    assert torch.allclose(out, f, atol=1e-6)
    l = torch.randn(2, 32, 8, 8)
    p = torch.ones(2, 1, 8, 8) * 0.5
    d = torch.zeros(2, 1, 8, 8)
    lg = ia(l, p, d, c)
    assert torch.allclose(lg, l, atol=1e-6)


def test_ct_zero_kills_texture_residual():
    m = build_asrt_v4()
    m.eval()
    low = _rand_img(1, 64, 64)
    ref = _rand_img(1, 64, 64)
    # Force C_T=0 by zeroing texture head bias very negative + weight
    with torch.no_grad():
        m.reliability.texture_head.weight.zero_()
        m.reliability.texture_head.bias.fill_(-20.0)
        m.reliability.illum_head.weight.zero_()
        m.reliability.illum_head.bias.fill_(-20.0)
    with torch.no_grad():
        out_ref = m(low, ref=ref)
        out_noref = m(low, ref=None)
    assert torch.allclose(out_ref, out_noref, atol=1e-4)


def test_ref_none_bypasses_matching():
    m = build_asrt_v4()
    m.eval()
    low = _rand_img(1, 64, 64)
    # Poison ref encoder — must not be called
    called = {'n': 0}
    orig = m.ref_encoder.forward

    def boom(x):
        called['n'] += 1
        return orig(x)

    m.ref_encoder.forward = boom
    with torch.no_grad():
        out = m(low, ref=None)
    assert called['n'] == 0
    assert out.shape == low.shape
    assert m.last_aux['used_reference'] is False


def test_no_lte_in_v4():
    m = build_asrt_v4()
    assert not hasattr(m, 'lte') or getattr(m, 'lte', None) is None
    names = [n for n, _ in m.named_modules()]
    assert not any('lte' == n.split('.')[0] for n, _ in m.named_children())
    assert isinstance(m.backbone.RB11[0], ResBlock)
    assert not any(type(rb).__name__ == 'CondResBlock' for rb in m.backbone.RB11)


def test_igab_unchanged_path():
    """Guided illum equals low illum when C_L=0 and zero-init adapter."""
    m = build_asrt_v4()
    m.eval()
    low = _rand_img(1, 64, 64)
    ref = _rand_img(1, 64, 64)
    with torch.no_grad():
        m.reliability.illum_head.weight.zero_()
        m.reliability.illum_head.bias.fill_(-20.0)
        m.reliability.texture_head.weight.zero_()
        m.reliability.texture_head.bias.fill_(-20.0)
        out_r = m(low, ref=ref)
        out_n = m(low, ref=None)
    assert torch.allclose(out_r, out_n, atol=1e-4)


def test_end_to_end_grad_after_steps():
    torch.manual_seed(0)
    m = build_asrt_v4()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    low = _rand_img(2, 64, 64)
    ref = _rand_img(2, 64, 64)
    gt = _rand_img(2, 64, 64)
    # Zero-init adapters: first step may have zero adapter-internal grads on
    # pre-last layers via C*delta; run several steps.
    hit = {k: False for k in (
        'matcher.query_proj', 'matcher.key_proj', 'ref_encoder',
        'reliability', 'tex_adapt', 'illum_adapt', 'backbone.SFE')}
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        pred = m(low, ref=ref)
        pred_n = m(low, ref=None)
        loss = (pred - gt).abs().mean() + 0.5 * (pred_n - gt).abs().mean()
        loss.backward()
        for name, p in m.named_parameters():
            if p.grad is None or p.grad.abs().sum() <= 0:
                continue
            for key in hit:
                if key in name:
                    hit[key] = True
        opt.step()
    missing = [k for k, v in hit.items() if not v]
    assert not missing, 'no grad reached: %s' % missing


def test_nonsquare_and_pad():
    m = build_asrt_v4()
    m.eval()
    low = _rand_img(1, 60, 84)
    ref = _rand_img(1, 60, 84)
    with torch.no_grad():
        out = m(low, ref=ref)
        out0 = m(low, ref=None)
    assert out.shape == low.shape
    assert out0.shape == low.shape


def test_stage_a_load_report():
    ckpt = '/root/data/experiments/ASRT/stage_a/checkpoints/best.pt'
    if not os.path.isfile(ckpt):
        return
    m = build_asrt_v4()
    ck = torch.load(ckpt, map_location='cpu')
    report = load_stage_a_into_v4(m, ck)
    assert report['loaded'] > 200
    assert any(k.startswith('lte.') for k in report['unused_ckpt'])
    assert any('ref_encoder' in k or 'matcher' in k for k in report['expected_new'])
    # All backbone params that exist in Stage A must be loaded
    assert not any(
        k.startswith('backbone.') and 'tex_adapt' not in k and 'illum_adapt' not in k
        for k in report['expected_new'])


def test_optimizer_groups_v4():
    m = build_asrt_v4()
    cfg = dict(stage='b_v4', lr_backbone=1e-5, lr_v4=1e-4, lr_igab=2e-5, lr_illum=2e-5,
               lr_inject=1e-5)
    opt = build_optimizer(m, cfg)
    names = {g['name'] for g in opt.param_groups}
    assert 'v4_new' in names
    assert 'backbone' in names


def test_dual_forward_smoke_cpu():
    m = build_asrt_v4()
    low = _rand_img(1, 64, 64)
    ref = _rand_img(1, 64, 64)
    gt = _rand_img(1, 64, 64)
    pred_r = m(low, ref)
    pred_n = m(low, None)
    loss = (pred_r - gt).abs().mean() + 0.5 * (pred_n - gt).abs().mean()
    loss.backward()
    assert pred_r.shape == low.shape


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print('running', name, '...')
            fn()
            print('  ok')
    print('ALL PASS')
