#!/usr/bin/env python
"""ASRT IRP v1 smoke: forward/backward, tiny overfit, A1/A2/A3, resource audit."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.models.factory import build_model  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402

OUT = Path('/root/data/experiments/ASRT/irp_v1_smoke')


def make_batch(b=2, h=128, w=128, device='cuda'):
    low = torch.rand(b, 3, h, w, device=device) * 2 - 1
    gt = torch.rand(b, 3, h, w, device=device) * 2 - 1
    ref = torch.rand(b, 3, h, w, device=device) * 2 - 1
    return low, gt, ref


def resource_probe(model, device, sizes=((128, 128), (256, 256))):
    rows = []
    model.eval()
    for h, w in sizes:
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        low, _, ref = make_batch(1, h, w, device)
        t0 = time.time()
        with torch.no_grad():
            _ = model(low, ref)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        fwd = time.time() - t0
        mem_fwd = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.startswith('cuda') else 0

        model.train()
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        low, gt, ref = make_batch(1, h, w, device)
        t0 = time.time()
        pred = model(low, ref)
        loss = F.mse_loss(pred, gt)
        loss.backward()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        bwd = time.time() - t0
        mem_bwd = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.startswith('cuda') else 0
        model.zero_grad(set_to_none=True)
        rows.append(dict(
            hw='%dx%d' % (h, w), fwd_sec=fwd, fwd_bwd_sec=bwd,
            peak_mem_fwd_mb=mem_fwd, peak_mem_bwd_mb=mem_bwd,
        ))
        print('resource', rows[-1], flush=True)
    return rows


def tiny_overfit(cfg, device, steps=30):
    seed_everything(42)
    model = build_asrt_irp(cfg).to(device)
    low, gt, ref = make_batch(2, 128, 128, device)
    # freeze one tiny batch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    model.train()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(low, ref)
        loss = F.mse_loss(pred, gt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if i % 10 == 0 or i == steps - 1:
            print('overfit step', i, 'loss', losses[-1], flush=True)
    return dict(losses=losses, dropped=losses[0] - losses[-1], ok=losses[-1] < losses[0])


def mode_checks(cfg, device):
    results = {}
    for src in ('none', 'low', 'nano'):
        c = dict(cfg)
        c['model'] = dict(cfg['model'])
        c['model']['prior_source'] = src
        m = build_asrt_irp(c).to(device)
        low, gt, ref = make_batch(2, 128, 128, device)
        y = m(low, ref if src == 'nano' else None)
        assert y.shape == low.shape
        assert torch.isfinite(y).all()
        # gradient on a named param
        loss = F.mse_loss(y, gt)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        results[src] = dict(shape=list(y.shape), finite=True, has_grad=has_grad,
                            params=m.count_parameters())
        print('mode', src, results[src]['params']['total'], 'grad', has_grad, flush=True)
    # A2 ignores external ref
    c = dict(cfg); c['model'] = dict(cfg['model']); c['model']['prior_source'] = 'low'
    m = build_asrt_irp(c).to(device)
    m.eval()
    low = torch.rand(1, 3, 128, 128, device=device) * 2 - 1
    r1 = torch.rand(1, 3, 128, 128, device=device) * 2 - 1
    r2 = torch.rand(1, 3, 128, 128, device=device) * 2 - 1
    with torch.no_grad():
        y1 = m(low, r1)
        y2 = m(low, r2)
        y0 = m(low, None)
    results['a2_ignores_ref'] = bool(torch.allclose(y1, y2, atol=1e-6))
    results['a2_noref_same'] = bool(torch.allclose(y1, y0, atol=1e-6))
    # A3 depends on ref
    c['model']['prior_source'] = 'nano'
    m3 = build_asrt_irp(c).to(device)
    m3.eval()
    with torch.no_grad():
        z1 = m3(low, r1)
        z2 = m3(low, r2)
        zn = m3(low, None)
    results['a3_depends_on_ref'] = float((z1 - z2).abs().max()) > 1e-8
    results['a3_ref_max_delta'] = float((z1 - z2).abs().max())
    results['a3_noref_runs'] = bool(torch.isfinite(zn).all())
    # A2/A3 param count equal
    results['a2_a3_param_equal'] = (
        results['low']['params']['total'] == results['nano']['params']['total'])
    # A3(ref=low) ≈ A2 with shared init
    seed_everything(0)
    m_a2 = build_asrt_irp({**cfg, 'model': {**cfg['model'], 'prior_source': 'low'}})
    seed_everything(0)
    m_a3 = build_asrt_irp({**cfg, 'model': {**cfg['model'], 'prior_source': 'nano'}})
    m_a2.load_state_dict(m_a3.state_dict())  # same weights; A2 ignores attn_ref? wait A2 has dual
    # Both have dual branches; with same weights A3(ref=low) should match A2(low)
    m_a2.eval(); m_a3.eval()
    low = torch.rand(1, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        ya2 = m_a2(low, None)
        ya3 = m_a3(low, low)
    results['a3_ref_low_matches_a2'] = bool(torch.allclose(ya2, ya3, atol=1e-5))
    # old ASRT still builds
    old = build_model({'stage': 'a', 'model': {'n_feats': 64, 'num_res_blocks': '8+8+4+2'}})
    results['old_asrt_builds'] = old is not None
    print('mode_checks', {k: results[k] for k in results if not isinstance(results[k], dict)
                          or k in ('a2_ignores_ref',)}, flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(ROOT) / 'configs' / 'asrt_irp_v1.yaml'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    device = a.device

    report = dict(device=device)
    report['modes'] = mode_checks(cfg, device)
    report['overfit'] = tiny_overfit(cfg, device)
    m = build_asrt_irp(cfg).to(device)
    report['params'] = m.count_parameters()
    report['resources'] = resource_probe(m, device)
    # AMP
    if device.startswith('cuda'):
        m = build_asrt_irp(cfg).to(device)
        low, gt, ref = make_batch(2, 128, 128, device)
        scaler = torch.cuda.amp.GradScaler()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred = m(low, ref)
            loss = F.mse_loss(pred, gt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        report['amp'] = dict(ok=True, loss=float(loss.item()))
    else:
        report['amp'] = dict(ok=False, reason='no cuda')

    path = OUT / 'smoke_report.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print('SMOKE DONE', path, flush=True)
    print(json.dumps({
        'overfit_ok': report['overfit']['ok'],
        'a2_a3_equal': report['modes']['a2_a3_param_equal'],
        'a3_depends': report['modes']['a3_depends_on_ref'],
        'params': report['params']['total'],
    }, indent=2))


if __name__ == '__main__':
    main()
