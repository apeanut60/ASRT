#!/usr/bin/env python
"""Evaluate ASRT checkpoint (RGB PSNR/SSIM)."""
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
from asrt.evaluation.inference import infer_full, infer_tiled  # noqa: E402
from asrt.evaluation.metrics import psnr_ssim_rgb  # noqa: E402
from asrt.models.asrt import build_asrt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--mode', choices=['correct', 'bypass', 'both', 'diag'], default='both')
    ap.add_argument('--tiled', action='store_true')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = {}
    if a.config:
        cfg = yaml.safe_load(open(a.config))
    model_cfg = dict(cfg.get('model', {}) or {})
    if cfg.get('query_alignment') is not None:
        model_cfg['query_align'] = dict(cfg['query_alignment'])
    model = build_asrt(model_cfg).to(device)
    ck = torch.load(a.ckpt, map_location=device)
    state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()
    rows = load_manifest(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    results = []
    for i, row in enumerate(rows):
        low = read_rgb_m11(row['low_path'])[None].to(device)
        gt = read_rgb_m11(row['gt_path'])[None].to(device)
        if tuple(gt.shape[-2:]) != tuple(low.shape[-2:]):
            gt = torch.nn.functional.interpolate(
                gt, size=low.shape[-2:], mode='bilinear', align_corners=False)
        ref = None
        if row.get('ref_path') and a.mode in ('correct', 'both', 'diag'):
            ref = read_rgb_m11(row['ref_path'])[None].to(device)
            if tuple(ref.shape[-2:]) != tuple(low.shape[-2:]):
                ref = torch.nn.functional.interpolate(
                    ref, size=low.shape[-2:], mode='bilinear', align_corners=False)
        rec = dict(sample_id=row['sample_id'])
        if a.mode in ('bypass', 'both'):
            pred = infer_tiled(model, low, None, a.tile) if a.tiled else infer_full(model, low, None)
            rec['bypass'] = psnr_ssim_rgb(pred, gt)
        if a.mode in ('correct', 'both') and ref is not None:
            pred = infer_tiled(model, low, ref, a.tile) if a.tiled else infer_full(model, low, ref)
            rec['correct'] = psnr_ssim_rgb(pred, gt)
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
        ckpt=a.ckpt, manifest=a.manifest, mode=a.mode, tiled=a.tiled,
        bypass=mean_key('bypass'), correct=mean_key('correct'), n=len(results))
    print(json.dumps(summary, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(dict(summary=summary, per_image=results), open(a.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
