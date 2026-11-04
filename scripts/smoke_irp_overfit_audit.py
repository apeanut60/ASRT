#!/usr/bin/env python
"""P1-5: Reproduce tiny-overfit spike and audit real-triplet overfit stability."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.irp_dataset import TripletIRPDataset, collate_irp  # noqa: E402
from asrt.data.manifest import load_manifest  # noqa: E402
from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402

OUT = Path('/root/data/experiments/ASRT/irp_v1_repair/overfit_audit')


def _log_step(model, loss, opt, step, rows):
    gn = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gn += float(p.grad.data.norm().item()) ** 2
    gn = gn ** 0.5
    bb = model.backbone
    with torch.no_grad():
        pred = rows.get('pred')
        rec = dict(
            step=step,
            loss=float(loss.item()),
            grad_norm=gn,
            lr=float(opt.param_groups[0]['lr']),
            alpha_h2=float(bb.middle_irp.alpha.item()),
            beta_h2=float(bb.middle_irp.beta.item()) if bb.middle_irp.beta is not None else None,
            alpha_h1=float(bb.fine_irp.alpha.item()),
            beta_h1=float(bb.fine_irp.beta.item()) if bb.fine_irp.beta is not None else None,
            pred_min=float(pred.min()) if pred is not None else None,
            pred_max=float(pred.max()) if pred is not None else None,
            pred_mean=float(pred.mean()) if pred is not None else None,
            scaler=float(rows.get('scaler', 1.0)),
        )
    return rec


def synth_overfit(cfg, device, lr, steps=30, amp=False):
    seed_everything(42)
    model = build_asrt_irp(cfg).to(device)
    low = torch.rand(2, 3, 128, 128, device=device) * 2 - 1
    gt = torch.rand(2, 3, 128, 128, device=device) * 2 - 1
    ref = torch.rand(2, 3, 128, 128, device=device) * 2 - 1
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.startswith('cuda'))
    logs = []
    model.train()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        if amp and device.startswith('cuda'):
            with torch.cuda.amp.autocast():
                pred = model(low, ref)
                loss = F.mse_loss(pred, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sc = float(scaler.get_scale())
        else:
            pred = model(low, ref)
            loss = F.mse_loss(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sc = 1.0
        logs.append(_log_step(model, loss, opt, i, dict(pred=pred.detach(), scaler=sc)))
        if i % 5 == 0 or i == steps - 1:
            print('synth', logs[-1], flush=True)
    return logs


def real_overfit(cfg, device, lr, steps=40, n_images=4):
    seed_everything(42)
    model = build_asrt_irp(cfg).to(device)
    man_root = Path(cfg['data']['manifest_root'])
    rows = load_manifest(str(man_root / 'lolv2_real_triplet_train.json'))[:n_images]
    ref_root = Path(cfg['data']['real_ref'])
    for r in rows:
        if not r.get('ref_path'):
            cand = ref_root / (str(r['sample_id']) + '.png')
            if cand.exists():
                r['ref_path'] = str(cand)
    ds = TripletIRPDataset(
        rows, crop=128, seed=42, augment=False, halo=0, require_ref=True,
        ref_mode='correct')
    # Fixed batch of all images (or first 2)
    lows, gts, refs = [], [], []
    for i in range(min(2, len(ds))):
        it = ds[i]
        lows.append(it['low'])
        gts.append(it['gt'])
        refs.append(it['ref'])
    low = torch.stack(lows).to(device)
    gt = torch.stack(gts).to(device)
    ref = torch.stack(refs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    logs = []
    model.train()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(low, ref)
        loss = F.mse_loss(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        logs.append(_log_step(model, loss, opt, i, dict(pred=pred.detach())))
        if i % 10 == 0 or i == steps - 1:
            print('real', logs[-1], flush=True)
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(ROOT) / 'configs' / 'asrt_irp_v1.yaml'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    report = {}
    # Reproduce smoke lr=1e-3 spike condition
    report['synth_lr_1e-3'] = synth_overfit(cfg, a.device, lr=1e-3, steps=30, amp=False)
    report['synth_lr_1e-4'] = synth_overfit(cfg, a.device, lr=1e-4, steps=30, amp=False)
    if a.device.startswith('cuda'):
        report['synth_lr_1e-4_amp'] = synth_overfit(
            cfg, a.device, lr=1e-4, steps=30, amp=True)
    try:
        report['real_lr_1e-4'] = real_overfit(cfg, a.device, lr=1e-4, steps=40)
    except Exception as e:
        report['real_lr_1e-4_error'] = str(e)
    # Summaries
    def summarize(logs):
        if not logs:
            return None
        losses = [x['loss'] for x in logs]
        return dict(
            first=losses[0], last=losses[-1],
            max=max(losses), dropped=losses[0] - losses[-1],
            spike=max(losses) > 2.0 * losses[0],
            max_grad=max(x['grad_norm'] for x in logs),
        )
    summary = {k: summarize(v) if isinstance(v, list) else v for k, v in report.items()}
    path = OUT / 'overfit_audit.json'
    with open(path, 'w') as f:
        json.dump(dict(summary=summary, logs=report), f, indent=2)
    print('OVERFIT AUDIT', json.dumps(summary, indent=2))
    print('wrote', path)


if __name__ == '__main__':
    main()
