#!/usr/bin/env python
"""Export per-image visual panels for LOLv2-Real Test (Stage A/B)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.manifest import load_manifest  # noqa: E402
from asrt.data.paired import read_rgb_m11  # noqa: E402
from asrt.evaluation.inference import infer_full  # noqa: E402
from asrt.evaluation.metrics import psnr_ssim_rgb, to_uint8_rgb  # noqa: E402
from asrt.models.factory import build_and_load  # noqa: E402


def load_model(config, ckpt, device):
    cfg = yaml.safe_load(open(config)) if config else {}
    model, report = build_and_load(cfg or {}, ckpt, device=device)
    model.eval()
    return model, int(report.get('step', -1))


def to_pil(t):
    return Image.fromarray(to_uint8_rgb(t, input_range='minus_one_one'))


def label_bar(width, text, height=28):
    bar = Image.new('RGB', (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(bar)
    draw.text((6, 6), text, fill=(240, 240, 240))
    return bar


def stack_h(images, labels):
    h = max(im.height for im in images)
    resized = []
    for im in images:
        if im.height != h:
            w = int(im.width * h / im.height)
            im = im.resize((w, h), Image.BICUBIC)
        resized.append(im)
    panels = []
    for im, lab in zip(resized, labels):
        bar = label_bar(im.width, lab)
        canvas = Image.new('RGB', (im.width, bar.height + im.height))
        canvas.paste(bar, (0, 0))
        canvas.paste(im, (0, bar.height))
        panels.append(canvas)
    total_w = sum(p.width for p in panels)
    out = Image.new('RGB', (total_w, panels[0].height), (0, 0, 0))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--config_a', default='configs/stage_a_lolv2_restore.yaml')
    ap.add_argument('--ckpt_a', default='/root/data/experiments/ASRT/stage_a/checkpoints/best.pt')
    ap.add_argument('--config_b', default='configs/stage_b_lolv2_reference.yaml')
    ap.add_argument('--ckpt_b', default='/root/data/experiments/ASRT/stage_b/checkpoints/best.pt')
    ap.add_argument('--b_label', default='B', help='Panel label prefix for stage-B model (e.g. Bv2)')
    ap.add_argument(
        '--manifest',
        default='/root/data/experiments/ASRT/manifests/lolv2_real_test_triplet.json')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = a.out_dir
    cases_dir = os.path.join(out_dir, 'cases')
    os.makedirs(cases_dir, exist_ok=True)
    bl = a.b_label

    model_a, step_a = load_model(a.config_a, a.ckpt_a, device)
    model_b, step_b = load_model(a.config_b, a.ckpt_b, device)

    rows = load_manifest(a.manifest)
    if a.limit:
        rows = rows[:a.limit]

    index = []
    for i, row in enumerate(rows):
        sid = row['sample_id']
        low = read_rgb_m11(row['low_path'])[None].to(device)
        gt = read_rgb_m11(row['gt_path'])[None].to(device)
        if tuple(gt.shape[-2:]) != tuple(low.shape[-2:]):
            gt = torch.nn.functional.interpolate(
                gt, size=low.shape[-2:], mode='bilinear', align_corners=False)
        ref = None
        if row.get('ref_path'):
            ref = read_rgb_m11(row['ref_path'])[None].to(device)
            if tuple(ref.shape[-2:]) != tuple(low.shape[-2:]):
                ref = torch.nn.functional.interpolate(
                    ref, size=low.shape[-2:], mode='bilinear', align_corners=False)

        with torch.no_grad():
            pred_a = infer_full(model_a, low, None)
            pred_b0 = infer_full(model_b, low, None)
            pred_b1 = infer_full(model_b, low, ref) if ref is not None else pred_b0

        m_a = psnr_ssim_rgb(pred_a, gt)
        m_b0 = psnr_ssim_rgb(pred_b0, gt)
        m_b1 = psnr_ssim_rgb(pred_b1, gt)

        imgs = [to_pil(low), to_pil(pred_a), to_pil(pred_b0), to_pil(pred_b1)]
        labels = [
            'Low',
            'A-bypass %.2fdB' % m_a['psnr_rgb'],
            '%s-bypass %.2fdB' % (bl, m_b0['psnr_rgb']),
            '%s-correct %.2fdB' % (bl, m_b1['psnr_rgb']),
        ]
        if ref is not None:
            imgs.append(to_pil(ref))
            labels.append('NanoRef')
        imgs.append(to_pil(gt))
        labels.append('GT')

        panel = stack_h(imgs, labels)
        case_dir = os.path.join(cases_dir, sid)
        os.makedirs(case_dir, exist_ok=True)
        panel_path = os.path.join(case_dir, 'panel.jpg')
        panel.save(panel_path, quality=92)
        # also save singles for zoom
        to_pil(pred_a).save(os.path.join(case_dir, 'pred_a_bypass.png'))
        to_pil(pred_b0).save(os.path.join(case_dir, 'pred_b_bypass.png'))
        to_pil(pred_b1).save(os.path.join(case_dir, 'pred_b_correct.png'))
        to_pil(low).save(os.path.join(case_dir, 'low.png'))
        to_pil(gt).save(os.path.join(case_dir, 'gt.png'))
        if ref is not None:
            to_pil(ref).save(os.path.join(case_dir, 'ref.png'))

        rec = dict(
            sample_id=sid,
            panel=panel_path,
            stage_a_bypass=m_a,
            stage_b_bypass=m_b0,
            stage_b_correct=m_b1,
            delta_b_correct_minus_a=m_b1['psnr_rgb'] - m_a['psnr_rgb'],
            delta_b_correct_minus_b_bypass=m_b1['psnr_rgb'] - m_b0['psnr_rgb'],
        )
        index.append(rec)
        json.dump(rec, open(os.path.join(case_dir, 'metrics.json'), 'w'), indent=2)
        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            print('%d/%d %s A=%.2f B0=%.2f B1=%.2f' % (
                i + 1, len(rows), sid, m_a['psnr_rgb'], m_b0['psnr_rgb'], m_b1['psnr_rgb']),
                flush=True)

    index_sorted = sorted(index, key=lambda r: r['delta_b_correct_minus_a'], reverse=True)
    payload = dict(
        n=len(index),
        stage_a_ckpt_step=step_a,
        stage_b_ckpt_step=step_b,
        config_b=a.config_b,
        ckpt_b=a.ckpt_b,
        b_label=bl,
        mean=dict(
            stage_a_bypass=float(np.mean([r['stage_a_bypass']['psnr_rgb'] for r in index])),
            stage_b_bypass=float(np.mean([r['stage_b_bypass']['psnr_rgb'] for r in index])),
            stage_b_correct=float(np.mean([r['stage_b_correct']['psnr_rgb'] for r in index])),
        ),
        best_ref_gain=index_sorted[:15],
        worst_ref_gain=list(reversed(index_sorted[-15:])),
        all=index,
    )
    json.dump(payload, open(os.path.join(out_dir, 'cases_index.json'), 'w'), indent=2)
    # HTML gallery for quick browse
    html = ['<html><head><meta charset="utf-8"><title>ASRT Test Cases</title>',
            '<style>body{font-family:sans-serif} img{max-width:100%;height:auto}'
            ' .card{margin:16px 0;padding:8px;border:1px solid #ddd}</style></head><body>',
            '<h1>LOLv2-Real Test — Stage A / %s visual cases</h1>' % bl,
            '<p>A best step %s | %s best step %s | n=%d</p>' % (step_a, bl, step_b, len(index)),
            '<p>Mean PSNR A-bypass=%.3f | %s-bypass=%.3f | %s-correct=%.3f</p>' % (
                payload['mean']['stage_a_bypass'], bl, payload['mean']['stage_b_bypass'],
                bl, payload['mean']['stage_b_correct']),
            '<h2>Sorted by (%s-correct − A-bypass) descending</h2>' % bl]
    for r in index_sorted:
        rel = os.path.relpath(r['panel'], out_dir)
        html.append('<div class="card"><h3>%s | Δ(%scorr-A)=%+.2f | A=%.2f B0=%.2f B1=%.2f</h3>' % (
            r['sample_id'], bl, r['delta_b_correct_minus_a'],
            r['stage_a_bypass']['psnr_rgb'], r['stage_b_bypass']['psnr_rgb'],
            r['stage_b_correct']['psnr_rgb']))
        html.append('<img src="%s" loading="lazy"></div>' % rel)
    html.append('</body></html>')
    open(os.path.join(out_dir, 'gallery.html'), 'w').write('\n'.join(html))
    print('wrote', os.path.join(out_dir, 'gallery.html'))
    print('wrote', os.path.join(out_dir, 'cases_index.json'))


if __name__ == '__main__':
    main()
