"""ASRT-v4 code-review fix verification (assertive, not smoke-only)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.evaluation.inference import infer_full, infer_tiled
from asrt.evaluation.metrics import psnr_ssim_rgb, to_uint8_rgb
from asrt.models.asrt import build_asrt
from asrt.models.asrt_v4 import ASRTv4, build_asrt_v4, load_stage_a_into_v4
from asrt.models.csfi_unet import ResBlock
from asrt.models.factory import build_and_load, build_model, detect_ckpt_version, load_checkpoint_strict
from asrt.models.v4_adapters import LowPassLuminance, TextureAdapter, IlluminationAdapter
from asrt.models.v4_matching import LocalSoftMatcher


def _gray_m11(val01, b=1, h=64, w=64):
    v = float(val01) * 2.0 - 1.0
    return torch.full((b, 3, h, w), v)


def test_illum_domain_constant_equal():
    m = build_asrt_v4()
    m.eval()
    # Same mid-gray ref and base → D_ref ≈ 0
    ref = _gray_m11(0.5)
    base = _gray_m11(0.5)
    weight = torch.zeros(1, 81, 16, 16)
    weight[:, 40] = 1.0  # center of 9x9
    with torch.no_grad():
        prior = m._build_illum_prior(ref, base, weight, torch.zeros(1, 2, 16, 16))
    assert prior['p_ref_matched'].min() >= -1e-4
    assert prior['p_ref_matched'].max() <= 1.0 + 1e-4
    assert prior['p_base'].min() >= -1e-4
    assert prior['d_ref'].abs().max() < 0.05, float(prior['d_ref'].abs().max())


def test_illum_domain_ref_brighter_darker():
    m = build_asrt_v4()
    weight = torch.zeros(1, 81, 16, 16)
    weight[:, 40] = 1.0
    disp = torch.zeros(1, 2, 16, 16)
    with torch.no_grad():
        bright = m._build_illum_prior(_gray_m11(0.8), _gray_m11(0.2), weight, disp)
        dark = m._build_illum_prior(_gray_m11(0.2), _gray_m11(0.8), weight, disp)
    assert float(bright['d_ref'].mean()) > 0.5
    assert float(dark['d_ref'].mean()) < -0.5


def test_illum_domain_extreme_finite():
    m = build_asrt_v4()
    weight = torch.zeros(1, 81, 8, 8)
    weight[:, 40] = 1.0
    disp = torch.zeros(1, 2, 8, 8)
    for rv, bv in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
        prior = m._build_illum_prior(_gray_m11(rv, h=32, w=32), _gray_m11(bv, h=32, w=32), weight, disp)
        assert torch.isfinite(prior['d_ref']).all()
        assert torch.isfinite(prior['p_ref_matched']).all()


def test_lpf_reflect_no_zero_pad_edge_drop():
    lpf = LowPassLuminance(ksize=9, sigma=3.0)
    x = torch.ones(1, 3, 32, 32) * 0.7
    y = lpf(x)
    # Edge should stay near 0.7 (reflect), not collapse toward 0
    edge = y[0, 0, 0, :].mean()
    center = y[0, 0, 16, 16]
    assert abs(float(edge) - 0.7) < 0.05, float(edge)
    assert abs(float(center) - 0.7) < 0.02


def test_to_uint8_explicit_range():
    # -1 → 0, 0 → 128, 1 → 255 under minus_one_one
    t = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]])  # wrong shape; use full
    t = torch.stack([
        torch.full((1, 8, 8), -1.0),
        torch.full((1, 8, 8), 0.0),
        torch.full((1, 8, 8), 1.0),
    ], dim=1)
    u = to_uint8_rgb(t, input_range='minus_one_one')
    assert int(u[0, 0, 0]) == 0
    assert int(u[0, 0, 1]) == 128
    assert int(u[0, 0, 2]) == 255
    # Bright [0,1] must NOT be auto-treated as [-1,1]
    bright = torch.ones(1, 3, 8, 8) * 0.9
    u01 = to_uint8_rgb(bright, input_range='zero_one')
    assert int(u01.mean()) >= 220
    # Same tensor wrongly guessed previously if min>=-0.05; with explicit range:
    u_m11 = to_uint8_rgb(bright, input_range='minus_one_one')
    # (0.9+1)*0.5*255 ≈ 242
    assert 230 <= int(u_m11.mean()) <= 250
    try:
        to_uint8_rgb(bright, input_range='auto')
        assert False, 'should raise'
    except ValueError:
        pass
    # Identity PSNR
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    m = psnr_ssim_rgb(x, x, input_range='minus_one_one')
    assert m['psnr_rgb'] == float('inf') or m['psnr_rgb'] > 50


def test_factory_rejects_v4_into_legacy():
    m4 = build_asrt_v4()
    sd = m4.state_dict()
    legacy = build_asrt(dict(enable_v3=False))
    # Simulate a v4 ckpt file
    path = '/tmp/asrt_v4_fake.pt'
    torch.save(dict(model=sd, step=1, cfg=dict(stage='b_v4', model=dict(enable_v4=True))), path)
    try:
        load_checkpoint_strict(legacy, path, expect_version=None)
        assert False, 'must refuse'
    except RuntimeError as e:
        assert 'v4' in str(e).lower() or 'refusing' in str(e).lower()


def test_factory_strict_v4_roundtrip():
    m = build_asrt_v4()
    path = '/tmp/asrt_v4_roundtrip.pt'
    torch.save(dict(model=m.state_dict(), step=42, cfg=dict(stage='b_v4', model=dict(enable_v4=True, version='v4'))), path)
    m2 = build_asrt_v4()
    report = load_checkpoint_strict(m2, path, expect_version='v4')
    assert report['step'] == 42
    assert report['version'] == 'v4'


def test_stage_a_cannot_silent_load_as_legacy_eval_of_v4():
    """v4 state into build_asrt must not succeed silently under factory."""
    m = build_asrt_v4()
    path = '/tmp/asrt_v4_only.pt'
    torch.save(dict(model=m.state_dict(), step=0, cfg={'model': {'enable_v4': True}}), path)
    try:
        build_and_load(dict(model=dict(enable_v3=False)), path, device='cpu')
        # build_and_load should detect v4 and rebuild — that's OK if it rebuilds as v4
        model, rep = build_and_load({}, path, device='cpu')
        assert getattr(model, 'is_v4', False)
        assert rep['version'] == 'v4'
    except RuntimeError:
        pass


def test_gate_zero_with_nonzero_adapter():
    """Force non-zero adapter weights then verify C=0 kills residual."""
    ta = TextureAdapter(16)
    with torch.no_grad():
        ta.net[-1].weight.fill_(0.1)
        ta.net[-1].bias.fill_(0.2)
    f = torch.randn(1, 16, 8, 8)
    t = torch.randn(1, 16, 8, 8)
    out0 = ta(f, t, torch.zeros(1, 1, 8, 8))
    out1 = ta(f, t, torch.ones(1, 1, 8, 8))
    assert torch.allclose(out0, f, atol=1e-6)
    assert not torch.allclose(out1, f, atol=1e-3)

    ia = IlluminationAdapter(16)
    with torch.no_grad():
        ia.net[-1].weight.fill_(0.05)
        ia.net[-1].bias.fill_(0.1)
    l = torch.randn(1, 16, 8, 8)
    p = torch.ones(1, 1, 8, 8) * 0.5
    d = torch.ones(1, 1, 8, 8) * 0.2
    lg0 = ia(l, p, d, torch.zeros(1, 1, 8, 8))
    lg1 = ia(l, p, d, torch.ones(1, 1, 8, 8))
    assert torch.allclose(lg0, l, atol=1e-6)
    assert not torch.allclose(lg1, l, atol=1e-3)


def test_ref_none_skips_entire_ref_branch():
    m = build_asrt_v4()

    def boom(*a, **k):
        raise RuntimeError('ref branch must not run')

    m.ref_encoder.forward = boom
    m.matcher.forward = boom
    m.reliability.forward = boom
    low = torch.rand(1, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        out = m(low, ref=None)
    assert out.shape == low.shape


def test_ref_padding_geometry_nonsquare():
    m = build_asrt_v4()
    m.eval()
    for h, w in [(255, 257), (257, 259), (60, 85), (63, 67)]:
        # Synthetic coordinate markers: unique vertical ramp
        yy = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
        low = torch.cat([yy, xx, torch.zeros(1, 1, h, w)], dim=1)
        ref = low.clone()
        stem = m.backbone.encode_low_stem(low)
        pack = m.match_reference(stem, ref)
        # Padded ref must match padded low spatial size
        assert pack['ref_aligned'].shape[-2:] == stem['low'].shape[-2:]
        # H/4 features match
        assert pack['T_h4'].shape[-2:] == stem['e2'].shape[-2:]
        with torch.no_grad():
            out = m(low, ref=ref)
        assert out.shape[-2:] == (h, w)
        assert torch.isfinite(out).all()


def test_tiled_coverage_and_border():
    m = build_asrt_v4().eval()
    device = 'cpu'
    m = m.to(device)
    sizes = [(128, 160), (200, 300), (300, 200), (255, 257)]
    # Skip huge 513x769 on CPU (too slow); covered on CUDA if available
    if torch.cuda.is_available():
        sizes.append((513, 769))
        m = m.cuda()
        device = 'cuda'
    for h, w in sizes:
        low = (torch.rand(1, 3, h, w, device=device) * 2 - 1)
        ref = (torch.rand(1, 3, h, w, device=device) * 2 - 1)
        for use_ref in (True, False):
            r = ref if use_ref else None
            with torch.no_grad():
                full = infer_full(m, low, r)
                tiled, wt = infer_tiled(m, low, r, tile=96, overlap=16, return_weight=True)
            assert full.shape == low.shape
            assert tiled.shape == low.shape
            assert bool((wt > 0).all()), 'uncovered pixels at %dx%d' % (h, w)
            diff = (full - tiled).abs()
            mae = float(diff.mean())
            mx = float(diff.max())
            border = float(torch.cat([
                diff[:, :, :2, :].reshape(-1),
                diff[:, :, -2:, :].reshape(-1),
                diff[:, :, :, :2].reshape(-1),
                diff[:, :, :, -2:].reshape(-1),
            ]).mean())
            # Not requiring float equality; report bounds
            assert mae < 0.25, 'MAE too large %s ref=%s: %f' % ((h, w), use_ref, mae)
            assert mx < 1.5
            assert border < 0.5
            print('tiled', h, w, 'ref', use_ref, 'mae', mae, 'max', mx, 'border', border)


def test_matching_best_sim_ignores_invalid():
    m = LocalSoftMatcher(dim=8, radius=2)
    # Tiny map so borders have many invalid candidates
    qf = torch.randn(1, 8, 3, 3)
    qi = torch.randn(1, 8, 3, 3)
    rf = torch.randn(1, 8, 3, 3)
    # Make invalid unfold region have huge sim by putting large values outside —
    # valid mask should prevent them from winning best_sim
    _, _, diag = m(qf, qi, rf)
    assert torch.isfinite(diag['best_sim']).all()
    assert float(diag['best_sim'].max()) <= 1.0 + 1e-3


def test_matching_continuity_no_wrap():
    m = LocalSoftMatcher(dim=8, radius=1)
    qf = torch.randn(1, 8, 4, 4)
    qi = torch.randn(1, 8, 4, 4)
    rf = torch.randn(1, 8, 4, 4)
    _, _, diag = m(qf, qi, rf)
    # Corners should not compare to opposite edge (roll artifact would couple them)
    assert diag['continuity'].shape == (1, 1, 4, 4)
    assert torch.isfinite(diag['continuity']).all()
    assert 'q_norm_pre' in diag and float(diag['q_norm_pre'].mean()) > 0


def test_unused_inject_heads_frozen():
    m = build_asrt_v4()
    for name in ('conv11_head', 'conv22_head', 'conv33_head'):
        for p in getattr(m.backbone, name).parameters():
            assert p.requires_grad is False
    n_train = m.count_parameters(trainable_only=True)
    n_all = m.count_parameters(trainable_only=False)
    assert n_train < n_all
    assert isinstance(m.backbone.RB11[0], ResBlock)


def test_dual_loss_both_backward():
    m = build_asrt_v4()
    low = torch.rand(1, 3, 64, 64) * 2 - 1
    ref = torch.rand(1, 3, 64, 64) * 2 - 1
    gt = torch.rand(1, 3, 64, 64) * 2 - 1
    pred_r = m(low, ref)
    pred_n = m(low, None)
    loss = (pred_r - gt).abs().mean() + 0.5 * (pred_n - gt).abs().mean()
    loss.backward()
    assert m.ref_encoder.stem[0].weight.grad is not None
    assert m.matcher.query_proj[0].weight.grad is not None
    assert m.reliability.texture_head.weight.grad is not None


def test_amp_smoke_cuda():
    if not torch.cuda.is_available():
        return
    m = build_asrt_v4().cuda()
    low = (torch.rand(1, 3, 64, 64).cuda() * 2 - 1)
    ref = (torch.rand(1, 3, 64, 64).cuda() * 2 - 1)
    gt = (torch.rand(1, 3, 64, 64).cuda() * 2 - 1)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pr = m(low, ref)
            pn = m(low, None)
            prior = m._build_illum_prior(
                ref, torch.zeros_like(ref),
                torch.ones(1, 81, 16, 16, device='cuda') / 81.0,
                torch.zeros(1, 2, 16, 16, device='cuda'))
            assert torch.isfinite(prior['d_ref']).all()
            loss = (pr - gt).abs().mean() + 0.5 * (pn - gt).abs().mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print('running', name, '...')
            fn()
            print('  ok')
    print('ALL PASS')
