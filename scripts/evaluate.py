#!/usr/bin/env python
"""Evaluate ASRT checkpoint (RGB PSNR/SSIM). Supports legacy / v3 / v4."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.manifest import load_manifest  # noqa: E402
from asrt.data.paired import read_rgb_m11  # noqa: E402
from asrt.evaluation.diagnostics import compare_ref_modes  # noqa: E402
from asrt.evaluation.metrics import psnr_ssim_rgb  # noqa: E402
from asrt.evaluation.protocol import infer_with_protocol, resolve_eval_cfg  # noqa: E402
from asrt.models.factory import build_and_load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--mode', choices=['correct', 'bypass', 'both', 'diag'], default='both')
    ap.add_argument('--tiled', action='store_true',
                    help='force tiled (overrides evaluation.inference_mode)')
    ap.add_argument('--full', action='store_true',
                    help='force full-image (overrides evaluation.inference_mode)')
    ap.add_argument('--tile', type=int, default=0)
    ap.add_argument('--overlap', type=int, default=-1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--allow_hparam_override', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = {}
    if a.config:
        cfg = yaml.safe_load(open(a.config)) or {}

    # Apply CLI inference overrides into cfg before resolve
    ev = dict(cfg.get('evaluation') or {})
    if a.tiled and a.full:
        raise SystemExit('pass only one of --tiled / --full')
    if a.tiled:
        ev['inference_mode'] = 'tiled'
    if a.full:
        ev['inference_mode'] = 'full'
    if a.tile > 0:
        ev['tile'] = a.tile
    if a.overlap >= 0:
        ev['overlap'] = a.overlap
    cfg['evaluation'] = ev
    eval_cfg = resolve_eval_cfg(cfg)

    model, load_report = build_and_load(
        cfg, a.ckpt, device=device,
        allow_hparam_override=bool(a.allow_hparam_override))
    model.eval()
    print(json.dumps(dict(
        model_version=load_report['version'],
        ckpt_step=load_report['step'],
        params=load_report['params'],
        trainable=load_report['trainable'],
        eval_cfg=eval_cfg,
        hparam_conflicts=load_report.get('hparam_conflicts'),
        hparam_note=load_report.get('hparam_note'),
    ), indent=2), flush=True)

    rows = load_manifest(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    results = []
    skipped_missing_ref = 0
    with torch.no_grad():
        for i, row in enumerate(rows):
            low = read_rgb_m11(row['low_path'])[None].to(device)
            gt = read_rgb_m11(row['gt_path'])[None].to(device)
            if tuple(gt.shape[-2:]) != tuple(low.shape[-2:]):
                gt = torch.nn.functional.interpolate(
                    gt, size=low.shape[-2:], mode='bilinear', align_corners=False)
            ref = None
            want_ref = a.mode in ('correct', 'both', 'diag')
            if want_ref and row.get('ref_path'):
                ref = read_rgb_m11(row['ref_path'])[None].to(device)
                if tuple(ref.shape[-2:]) != tuple(low.shape[-2:]):
                    ref = torch.nn.functional.interpolate(
                        ref, size=low.shape[-2:], mode='bilinear', align_corners=False)
            elif want_ref and a.mode in ('correct', 'diag'):
                skipped_missing_ref += 1
                continue
            rec = dict(sample_id=row.get('sample_id') or row.get('id'))
            if a.mode in ('bypass', 'both'):
                pred, meta = infer_with_protocol(model, low, None, eval_cfg=cfg)
                rec['bypass'] = psnr_ssim_rgb(
                    pred, gt, input_range=eval_cfg['input_range'])
                rec['bypass_meta'] = meta
            if a.mode in ('correct', 'both') and ref is not None:
                pred, meta = infer_with_protocol(model, low, ref, eval_cfg=cfg)
                rec['correct'] = psnr_ssim_rgb(
                    pred, gt, input_range=eval_cfg['input_range'])
                rec['correct_meta'] = meta
            if a.mode == 'diag' and ref is not None:
                rec['diag'] = compare_ref_modes(model, low, gt, ref, device)
            results.append(rec)
            if (i + 1) % 20 == 0:
                print(i + 1, flush=True)

    def mean_key(key):
        vals = [r[key]['psnr_rgb'] for r in results if key in r]
        ssims = [r[key]['ssim_rgb'] for r in results if key in r]
        if not vals:
            return None
        return dict(psnr_rgb=sum(vals) / len(vals), ssim_rgb=sum(ssims) / len(ssims), n=len(vals))

    summary = dict(
        ckpt=a.ckpt, manifest=a.manifest, mode=a.mode,
        eval_cfg=eval_cfg, load=load_report,
        bypass=mean_key('bypass'), correct=mean_key('correct'),
        n=len(results), skipped_missing_ref=skipped_missing_ref,
        sample_ids=[r.get('sample_id') for r in results],
    )
    print(json.dumps(summary, indent=2, default=str))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(dict(summary=summary, per_image=results), open(a.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
