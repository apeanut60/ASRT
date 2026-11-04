#!/usr/bin/env python
"""ASRT IRP v1 training entry: smoke wiring + formal IRPTrainer.

Examples:
  # Engineering smoke (keeps old short loop):
  python scripts/train_irp.py --config configs/asrt_irp_v1.yaml --smoke_steps 20

  # Formal (requires training.max_steps in config or --max_steps):
  python scripts/train_irp.py --config configs/asrt_irp_v1.yaml --max_steps 1000
  python scripts/train_irp.py --resume /path/to/checkpoint.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.irp_dataset import TripletIRPDataset, collate_irp  # noqa: E402
from asrt.data.manifest import load_manifest  # noqa: E402
from asrt.data.reference_modes import apply_reference_mode  # noqa: E402
from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.irp_trainer import IRPTrainer, validate_irp_cfg  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402


def _rows_from_cfg(cfg):
    man_root = Path(cfg.get('data', {}).get(
        'manifest_root', '/root/data/experiments/ASRT/manifests'))
    trip = man_root / 'lolv2_real_triplet_train.json'
    man = load_manifest(str(trip))
    rows = man if isinstance(man, list) else man.get('rows', man)
    ref_root = Path(cfg['data']['real_ref'])
    for r in rows:
        if not r.get('ref_path'):
            cand = ref_root / (str(r['sample_id']) + '.png')
            if not cand.exists():
                cand = ref_root / Path(r.get('low_path', 'x')).name
            if cand.exists():
                r['ref_path'] = str(cand)
    return rows


def run_smoke(cfg, device, steps: int, ref_mode: str):
    seed_everything(int(cfg.get('training', {}).get('seed', cfg.get('seed', 42))))
    model = build_asrt_irp(cfg).to(device)
    print('params', model.count_parameters(), flush=True)
    rows = _rows_from_cfg(cfg)[: max(8, int(cfg.get('training', {}).get('batch_size', 2)) * 2)]
    tr = cfg.get('training', {})
    ds = TripletIRPDataset(
        rows, crop=int(tr.get('crop_size', 128)), seed=int(tr.get('seed', 42)),
        halo=0, require_ref=(str(cfg.get('model', {}).get('prior_source', 'nano')) == 'nano'),
        ref_mode=ref_mode)
    loader = DataLoader(
        ds, batch_size=min(2, len(ds)), shuffle=True, num_workers=0,
        drop_last=False, collate_fn=collate_irp)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(tr.get('learning_rate', 1e-4)))
    it = iter(loader)
    model.train()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        low = batch['low'].to(device)
        gt = batch['gt'].to(device)
        ref = batch['ref']
        if ref is not None:
            ref = ref.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(low, ref)
        loss = F.mse_loss(pred, gt)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == 1:
            print('smoke step', step, 'loss', float(loss.item()), flush=True)
    print('DONE smoke_irp steps', steps, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(ROOT) / 'configs' / 'asrt_irp_v1.yaml'))
    ap.add_argument('--device', default=None)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--smoke_steps', type=int, default=0)
    ap.add_argument('--ref_mode', default=None)
    ap.add_argument('--resume', default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.max_steps is not None:
        cfg.setdefault('training', {})['max_steps'] = int(a.max_steps)
    if a.device:
        cfg.setdefault('training', {})['device'] = a.device
    ref_mode = a.ref_mode or str(cfg.get('data', {}).get('ref_mode', 'correct'))
    cfg.setdefault('data', {})['ref_mode'] = ref_mode

    # Explicit smoke path (engineering wiring; not formal training).
    if a.smoke_steps and not a.resume and a.max_steps is None:
        device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        cfg.setdefault('training', {})['allow_non_correct_ref'] = True
        print('SMOKE MODE ref_mode=%s (diagnostic; not formal training)' % ref_mode,
              flush=True)
        run_smoke(cfg, device, a.smoke_steps, ref_mode)
        return

    max_steps = cfg.get('training', {}).get('max_steps')
    if not a.resume and (max_steps is None or int(max_steps) <= 0):
        raise SystemExit(
            'Pass --smoke_steps N for wiring check, or set training.max_steps / '
            '--max_steps for formal IRPTrainer (full 80k not started in repair task).')

    # Formal training: ref_mode must be correct (eval interventions are separate).
    if str(ref_mode).lower().replace('-', '_') not in ('correct',):
        raise SystemExit(
            'Formal IRP training rejects --ref_mode=%r / data.ref_mode=%r. '
            'Use data.ref_mode=correct. Wrong/Shuffle/NoRef/Ref=Low are '
            'evaluation-only interventions (or pass --smoke_steps for diagnostics).'
            % (a.ref_mode, ref_mode))

    device = a.device or cfg.get('training', {}).get(
        'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    cfg.setdefault('training', {})['device'] = device
    if a.resume and (max_steps is None or int(max_steps) <= 0):
        # Resume requires a finite schedule horizon for the cosine scheduler.
        cfg['training']['max_steps'] = int(a.max_steps or 80000)
    trainer = IRPTrainer(cfg, device=device)
    if a.resume:
        print('resume', a.resume, flush=True)
        trainer.load_checkpoint(a.resume)
    if a.max_steps is not None:
        trainer.max_steps = int(a.max_steps)
        trainer.sched.total = int(a.max_steps)
    result = trainer.fit()
    print('DONE formal train_irp', result, flush=True)


if __name__ == '__main__':
    main()
