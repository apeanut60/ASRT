#!/usr/bin/env python
"""Paired A2/A3 validation comparison + optional A3 reference diagnostics.

Checkpoint loading uses factory.load_checkpoint_strict so prior_source and
other IRP hparams cannot silently mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asrt.data.irp_dataset import TripletIRPDataset  # noqa: E402
from asrt.data.reference_modes import build_wrong_mapping  # noqa: E402
from asrt.evaluation.metrics import psnr_ssim_rgb  # noqa: E402
from asrt.evaluation.protocol import infer_with_protocol, resolve_eval_cfg  # noqa: E402
from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.models.factory import load_checkpoint_strict  # noqa: E402
from asrt.training.irp_trainer import _load_rows  # noqa: E402


def _bootstrap_ci(deltas, n_boot=2000, seed=0):
    """Bootstrap over *images* (paired deltas), not over training checkpoints."""
    rng = np.random.RandomState(seed)
    deltas = np.asarray(deltas, dtype=np.float64)
    if len(deltas) == 0:
        return dict(mean=None, ci95=(None, None), median=None)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        means.append(sample.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(deltas.mean()), ci95=(float(lo), float(hi)),
                median=float(np.median(deltas)))


def _load_model(cfg_path, ckpt_path, device, expect_prior=None):
    cfg = yaml.safe_load(open(cfg_path))
    if expect_prior is not None:
        got = str(cfg['model']['prior_source']).lower()
        if got != str(expect_prior).lower():
            raise RuntimeError(
                'config prior_source=%r does not match expect %r' % (got, expect_prior))
    model = build_asrt_irp(cfg).to(device)
    report = load_checkpoint_strict(
        model, ckpt_path, device='cpu', expect_version='irp_v1',
        external_cfg=cfg, allow_hparam_override=False)
    model.eval()
    ck = torch.load(ckpt_path, map_location='cpu')
    return model, cfg, ck, report


@torch.no_grad()
def eval_rows(model, rows, cfg, device, ref_mode='correct', wrong_map=None):
    ds = TripletIRPDataset(
        rows, crop=0, seed=0, augment=False, halo=0,
        require_ref=(str(cfg['model']['prior_source']) == 'nano'),
        ref_mode=ref_mode, wrong_mapping=wrong_map,
        id_to_row={str(r['sample_id']): r for r in rows},
    )
    ev = resolve_eval_cfg(cfg)
    out = []
    for i in range(len(ds)):
        b = ds[i]
        low = b['low'][None].to(device)
        gt = b['gt'][None].to(device)
        ref = b['ref']
        if ref is not None:
            ref = ref[None].to(device)
        if ref_mode == 'noref':
            pred, meta = infer_with_protocol(model, low, None, cfg)
        else:
            pred, meta = infer_with_protocol(model, low, ref, cfg)
        m = psnr_ssim_rgb(pred, gt, input_range=str(ev.get('input_range', 'minus_one_one')))
        out.append(dict(
            sample_id=str(b['sample_id']),
            psnr=float(m['psnr_rgb']), ssim=float(m['ssim_rgb']),
            inference_used=meta.get('inference_used'),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a2_cfg', default=str(ROOT / 'configs/asrt_irp_v1_a2.yaml'))
    ap.add_argument('--a3_cfg', default=str(ROOT / 'configs/asrt_irp_v1_a3.yaml'))
    ap.add_argument('--a2_ckpt', required=True)
    ap.add_argument('--a3_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--diagnostics', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    m2, cfg2, ck2, rep2 = _load_model(a.a2_cfg, a.a2_ckpt, a.device, expect_prior='low')
    m3, cfg3, ck3, rep3 = _load_model(a.a3_cfg, a.a3_ckpt, a.device, expect_prior='nano')
    rows = _load_rows(cfg3, 'val')
    r2 = eval_rows(m2, rows, cfg2, a.device, ref_mode='correct')
    r3 = eval_rows(m3, rows, cfg3, a.device, ref_mode='correct')
    by2 = {x['sample_id']: x for x in r2}
    by3 = {x['sample_id']: x for x in r3}
    paired = []
    for sid in sorted(by2):
        a2, a3 = by2[sid], by3[sid]
        paired.append(dict(
            sample_id=sid,
            A2_PSNR=a2['psnr'], A3_PSNR=a3['psnr'],
            Delta_PSNR=a3['psnr'] - a2['psnr'],
            A2_SSIM=a2['ssim'], A3_SSIM=a3['ssim'],
            Delta_SSIM=a3['ssim'] - a2['ssim'],
        ))
    dpsnr = [p['Delta_PSNR'] for p in paired]
    boot = _bootstrap_ci(dpsnr)
    summary = dict(
        n_images=len(paired),
        mean_a2_psnr=float(np.mean([p['A2_PSNR'] for p in paired])),
        mean_a3_psnr=float(np.mean([p['A3_PSNR'] for p in paired])),
        mean_a2_ssim=float(np.mean([p['A2_SSIM'] for p in paired])),
        mean_a3_ssim=float(np.mean([p['A3_SSIM'] for p in paired])),
        mean_delta_psnr=boot['mean'],
        median_delta_psnr=boot['median'],
        bootstrap_ci95_delta_psnr=boot['ci95'],
        bootstrap_unit='validation_images',
        positive_delta_ratio=float(np.mean([d > 0 for d in dpsnr])),
        negative_delta_ratio=float(np.mean([d < 0 for d in dpsnr])),
        worst10_negative=sorted(paired, key=lambda x: x['Delta_PSNR'])[:10],
        a2_ckpt=a.a2_ckpt, a3_ckpt=a.a3_ckpt,
        a2_global_step=ck2.get('global_step'), a3_global_step=ck3.get('global_step'),
        a2_load_report=rep2, a3_load_report=rep3,
    )
    with open(out / 'paired_val_metrics.json', 'w') as f:
        json.dump(dict(summary=summary, per_image=paired), f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in summary if 'report' not in k}, indent=2))

    if a.diagnostics:
        ids = [str(r['sample_id']) for r in rows]
        wmap_path = out / 'wrong_ref_mapping_val.json'
        wmap = build_wrong_mapping(ids, seed=42, out_path=str(wmap_path))
        modes = ['correct', 'wrong', 'patch_shuffle', 'pixel_shuffle', 'ref_low', 'noref']
        diag = {}
        for mode in modes:
            diag[mode] = eval_rows(
                m3, rows, cfg3, a.device, ref_mode=mode,
                wrong_map=wmap if mode == 'wrong' else None)
            print('diag', mode,
                  float(np.mean([x['psnr'] for x in diag[mode]])), flush=True)
        by_c = {x['sample_id']: x for x in diag['correct']}
        deltas = {}
        for mode in modes:
            if mode == 'correct':
                continue
            by_m = {x['sample_id']: x for x in diag[mode]}
            d = [by_c[s]['psnr'] - by_m[s]['psnr'] for s in by_c]
            deltas[mode] = dict(
                mean_correct_minus_mode=float(np.mean(d)),
                per_image=[{'sample_id': s,
                            'delta': by_c[s]['psnr'] - by_m[s]['psnr']}
                           for s in sorted(by_c)],
            )
        with open(out / 'a3_reference_diagnostics.json', 'w') as f:
            json.dump(dict(modes={k: diag[k] for k in modes}, deltas=deltas),
                      f, indent=2)


if __name__ == '__main__':
    main()
