#!/usr/bin/env python
"""Training entry for ASRT IRP v1 (formal schedule frozen later).

This script wires model + MSE + AdamW. Do not start large-scale training here
until architecture smoke tests pass and a training protocol is frozen.
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

from asrt.data.manifest import load_manifest  # noqa: E402
from asrt.data.paired import PairedRestoreDataset, read_rgb_m11, sync_geometry  # noqa: E402
from asrt.data.reference_modes import apply_reference_mode  # noqa: E402
from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402


class TripletIRPDataset(PairedRestoreDataset):
    """Low/GT/Ref crops; returns [-1,1] tensors."""

    def __getitem__(self, idx):
        row = self.rows[idx]
        low = read_rgb_m11(row['low_path'])
        gt = read_rgb_m11(row['gt_path'])
        ref_path = row.get('ref_path')
        ref = read_rgb_m11(ref_path) if ref_path else None
        if tuple(gt.shape[-2:]) != tuple(low.shape[-2:]):
            gt = torch.nn.functional.interpolate(
                gt[None], size=low.shape[-2:], mode='bilinear', align_corners=False)[0]
        if ref is not None and tuple(ref.shape[-2:]) != tuple(low.shape[-2:]):
            ref = torch.nn.functional.interpolate(
                ref[None], size=low.shape[-2:], mode='bilinear', align_corners=False)[0]
        import numpy as np
        rng = np.random.RandomState(self.seed + self.epoch * 1000003 + idx)
        if self.augment and self.crop > 0:
            tensors = [low, gt] + ([ref] if ref is not None else [])
            out = sync_geometry(tensors, self.crop, rng)
            low, gt = out[0], out[1]
            ref = out[2] if ref is not None else None
        return dict(
            low=low, gt=gt, ref=ref,
            sample_id=row.get('sample_id', str(idx)),
            has_ref=ref is not None,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(ROOT) / 'configs' / 'asrt_irp_v1.yaml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_steps', type=int, default=0,
                    help='0 = refuse (formal training not started in impl task)')
    ap.add_argument('--smoke_steps', type=int, default=0,
                    help='optional tiny train loop for wiring check')
    ap.add_argument('--ref_mode', default=None,
                    help='correct|wrong|shuffle|ref_low|noref (overrides config)')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if not a.max_steps and not a.smoke_steps:
        raise SystemExit(
            'Formal IRP training not started. Pass --smoke_steps N for a wiring check, '
            'or freeze a training protocol then pass --max_steps.')

    seed_everything(int(cfg.get('training', {}).get('seed', cfg.get('seed', 42))))
    model = build_asrt_irp(cfg).to(a.device)
    ref_mode = a.ref_mode or str(cfg.get('data', {}).get('ref_mode', 'correct'))
    print('params', model.count_parameters(), flush=True)

    # Prefer real triplet manifest if present
    man_root = Path(cfg.get('data', {}).get(
        'manifest_root', '/root/data/experiments/ASRT/manifests'))
    trip = man_root / 'lolv2_real_triplet_train.json'
    if trip.exists():
        rows = load_manifest(str(trip))
    else:
        # fallback: paired + attach nano ref by id
        rows = load_manifest(str(man_root / 'lolv2_real_paired_train.json'))
        ref_root = Path(cfg['data']['real_ref'])
        for r in rows:
            cand = ref_root / (r['sample_id'] + '.png')
            r['ref_path'] = str(cand) if cand.exists() else None
        rows = [r for r in rows if r.get('ref_path')]

    tr = cfg.get('training', {})
    ds = TripletIRPDataset(
        rows, crop=int(tr.get('crop_size', 128)), seed=int(tr.get('seed', 42)))
    loader = DataLoader(
        ds, batch_size=int(tr.get('batch_size', 8)), shuffle=True,
        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr.get('learning_rate', 1e-4)),
        weight_decay=float(tr.get('weight_decay', 1e-4)))
    steps = a.smoke_steps or a.max_steps
    it = iter(loader)
    model.train()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        low = batch['low'].to(a.device)
        gt = batch['gt'].to(a.device)
        ref = batch['ref']
        if ref is not None:
            ref = ref.to(a.device)
        ref = apply_reference_mode(low, ref, mode=ref_mode)
        opt.zero_grad(set_to_none=True)
        pred = model(low, ref)
        loss = F.mse_loss(pred, gt)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == 1:
            print('step', step, 'loss', float(loss.item()), flush=True)
    print('DONE train_irp steps', steps, flush=True)


if __name__ == '__main__':
    main()
