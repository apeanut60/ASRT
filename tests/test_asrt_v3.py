"""ASRT-v3 functional acceptance tests."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt import ASRT, build_asrt, load_b_v2_into_v3
from asrt.models.cond_resblock import CondResBlock
from asrt.models.csfi_unet import CSFIUNet
from asrt.models.illumination_prior import IlluminationAdapter, build_matched_illumination
from asrt.models.reference import SearchTransfer
from asrt.training.optimizer import build_optimizer


def _rand_img(b=2, h=64, w=64):
    return (torch.rand(b, 3, h, w) * 2 - 1)


def test_search_transfer_diag_shapes():
    st = SearchTransfer()
    q = torch.randn(2, 256, 16, 16)
    k = torch.randn(2, 256, 16, 16)
    v1 = torch.randn(2, 64, 64, 64)
    v2 = torch.randn(2, 128, 32, 32)
    v3 = torch.randn(2, 256, 16, 16)
    S, T3, T2, T1, diag = st(q, k, v1, v2, v3, return_diag=True)
    assert S.shape == (2, 1, 16, 16)
    assert diag['match_index'].shape == (2, 256)
    assert diag['match_score'].shape == S.shape
    assert diag['match_margin'].shape == S.shape
    assert diag['cycle_consistency'].shape == S.shape
    assert diag['spatial_consistency'].shape == S.shape
    # backward compat
    S2, _, _, _ = st(q, k, v1, v2, v3, return_diag=False)
    assert torch.allclose(S, S2, atol=1e-5)


def test_ref_none_bypasses_v3():
    m = build_asrt(dict(enable_v3=True, freeze_lte=True))
    m.eval()
    low = _rand_img(1, 64, 64)
    with torch.no_grad():
        out0 = m(low, ref=None)
        out1 = m(low, ref=None)
    assert out0.shape == low.shape
    assert torch.allclose(out0, out1)
    assert m.last_aux['used_reference'] is False
    assert m.last_aux.get('v3') is None


def test_zero_init_illum_adapter_identity():
    ad = IlluminationAdapter(64)
    l = torch.randn(2, 64, 16, 16)
    ones = torch.ones(2, 1, 16, 16)
    guided, delta = ad(l, ones * 0.5, ones * 0.4, ones * 0.1, ones)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)
    assert torch.allclose(guided, l, atol=1e-6)


def test_zero_init_cond_resblock_matches_plain():
    from asrt.models.csfi_unet import ResBlock
    torch.manual_seed(0)
    plain = ResBlock(32)
    cond = CondResBlock(32)
    cond.conv1.load_state_dict(plain.conv1.state_dict())
    cond.conv2.load_state_dict(plain.conv2.state_dict())
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        y0 = plain(x)
        y1 = cond(x, None, None)
        lg = torch.randn(2, 32, 16, 16)
        cl = torch.ones(2, 1, 16, 16)
        y2 = cond(x, lg, cl)  # adapter zero → same
    assert torch.allclose(y0, y1, atol=1e-5)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_ct_zero_kills_texture_residual():
    bb = CSFIUNet(enable_v3=True, n_feats=64)
    bb.eval()
    low = _rand_img(1, 64, 64)
    S = torch.ones(1, 1, 16, 16)
    T3 = torch.randn(1, 256, 16, 16)
    T2 = torch.randn(1, 128, 32, 32)
    T1 = torch.randn(1, 64, 64, 64)
    c0 = torch.zeros(1, 1, 16, 16)
    c1 = torch.ones(1, 1, 16, 16)
    prior = dict(
        p_ref_matched=torch.ones(1, 1, 16, 16) * 0.5,
        p_base=torch.ones(1, 1, 16, 16) * 0.4,
        d_ref=torch.zeros(1, 1, 16, 16),
    )
    with torch.no_grad():
        out_ct0 = bb(low, S=S, T_lv3=T3, T_lv2=T2, T_lv1=T1,
                     use_reference=True, c_t=c0, c_l=c0, illum_prior=prior)
        diag = bb.last_v3_diag
        assert diag is not None
        # C_T=0 ⇒ texture residual strictly zero at all scales
        assert diag['inject_res_norm']['coarse'] < 1e-6
        assert diag['inject_res_norm']['middle'] < 1e-6
        assert diag['inject_res_norm']['fine'] < 1e-6
        _ = out_ct0, c1


def test_cl_zero_kills_illum_delta():
    ad = IlluminationAdapter(32)
    # Non-zero weights artificially
    with torch.no_grad():
        ad.conv2.weight.fill_(0.01)
        ad.conv2.bias.fill_(0.01)
    l = torch.randn(1, 32, 8, 8)
    ones = torch.ones(1, 1, 8, 8)
    guided, delta = ad(l, ones * 0.5, ones * 0.4, ones * 0.2, torch.zeros_like(ones))
    assert torch.allclose(guided, l, atol=1e-6)
    assert not torch.allclose(delta, torch.zeros_like(delta))  # delta computed but gated off


def test_match_index_warp_grid():
    ref = torch.zeros(1, 3, 64, 64)
    # paint unique luminance per H/4 cell via channels
    for i in range(16):
        for j in range(16):
            ref[0, :, i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = (i * 16 + j) / 255.0 * 2 - 1
    # identity match
    idx = torch.arange(16 * 16).view(1, -1)
    base = torch.zeros_like(ref)
    out = build_matched_illumination(ref, base, idx, 16, 16, lpf_k=1)
    assert out['p_ref_matched'].shape == (1, 1, 16, 16)
    assert out['d_ref'].shape == (1, 1, 16, 16)
    # global mean removed
    assert float(out['d_ref'].mean().abs()) < 1e-4


def test_v3_forward_shapes_and_grad():
    m = build_asrt(dict(enable_v3=True, freeze_lte=True, query_align=dict(enabled=False)))
    m.train()
    low = _rand_img(2, 64, 64)
    ref = _rand_img(2, 64, 64)
    out = m(low, ref=ref)
    assert out.shape == low.shape
    loss = out.mean()
    loss.backward()
    # New modules must receive grad
    assert m.reliability.head_t.weight.grad is not None
    assert m.backbone.coarse_illum_adapter.conv1.weight.grad is not None
    assert any(p.grad is not None for p in m.backbone.RB11[-1].adapter.parameters())


def test_load_b_v2_allows_only_new_missing():
    # Build v2-like then v3
    v2 = build_asrt(dict(enable_v3=False, freeze_lte=True))
    v3 = build_asrt(dict(enable_v3=True, freeze_lte=True))
    info = load_b_v2_into_v3(v3, v2.state_dict())
    assert info['loaded'] > 0
    assert len(info['missing']) > 0
    assert all(
        k.startswith('reliability.')
        or 'illum_adapter' in k
        or 'fine_ct_refine' in k
        or '.adapter.' in k
        for k in info['missing']
    )


def test_optimizer_covers_all_trainable():
    m = build_asrt(dict(enable_v3=True, freeze_lte=True))
    cfg = dict(stage='b', lr_backbone=1e-5, lr_inject=1e-4, lr_igab=2e-5,
               lr_illum=2e-5, lr_v3=1e-4)
    opt = build_optimizer(m, cfg)
    names = [g['name'] for g in opt.param_groups]
    assert 'v3_new' in names
    assert 'backbone' in names


def test_fp16_no_nan():
    if not torch.cuda.is_available():
        return
    m = build_asrt(dict(enable_v3=True, freeze_lte=True)).cuda().train()
    low = _rand_img(2, 64, 64).cuda()
    ref = _rand_img(2, 64, 64).cuda()
    with torch.cuda.amp.autocast():
        out = m(low, ref=ref)
        loss = out.abs().mean()
    assert torch.isfinite(out).all()
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


if __name__ == '__main__':
    tests = [n for n, f in globals().items() if n.startswith('test_') and callable(f)]
    failed = []
    for n in tests:
        try:
            print('RUN', n, flush=True)
            globals()[n]()
            print(' OK', n, flush=True)
        except Exception as e:
            print(' FAIL', n, e, flush=True)
            failed.append((n, e))
    if failed:
        raise SystemExit('%d failed' % len(failed))
    print('ALL_PASS', len(tests))
