#!/usr/bin/env python
"""Plot A2/A3 training / validation curves from experiment logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def _read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a2_dir', required=True)
    ap.add_argument('--a3_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logs = {
        'A2': _read_jsonl(Path(a.a2_dir) / 'train_log.jsonl'),
        'A3': _read_jsonl(Path(a.a3_dir) / 'train_log.jsonl'),
    }
    vals = {
        'A2': _read_jsonl(Path(a.a2_dir) / 'val_log.jsonl'),
        'A3': _read_jsonl(Path(a.a3_dir) / 'val_log.jsonl'),
    }

    def _plot_metric(key, fname, ylabel, source='train'):
        plt.figure(figsize=(8, 4.5))
        for name, color in (('A2', '#1f77b4'), ('A3', '#d62728')):
            rows = logs[name] if source == 'train' else vals[name]
            xs = [r.get('global_step', r.get('step')) for r in rows if key in r]
            ys = [r[key] for r in rows if key in r]
            if xs:
                plt.plot(xs, ys, label=name, color=color, linewidth=1.5)
        plt.xlabel('Optimizer Updates')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / fname, dpi=140)
        plt.close()

    _plot_metric('loss', 'training_loss_a2_a3.png', 'Train Loss')
    _plot_metric('learning_rate', 'learning_rate_curve.png', 'Learning Rate')
    _plot_metric('psnr', 'validation_psnr_a2_a3.png', 'Val PSNR', source='val')
    _plot_metric('ssim', 'validation_ssim_a2_a3.png', 'Val SSIM', source='val')

    # alpha / beta
    plt.figure(figsize=(8, 4.5))
    for name, color in (('A2', '#1f77b4'), ('A3', '#d62728')):
        rows = logs[name]
        xs = [r['global_step'] for r in rows]
        for key, ls in (('alpha_h2', '-'), ('beta_h2', '--'),
                        ('alpha_h1', ':'), ('beta_h1', '-.')):
            ys = [r.get(key) for r in rows]
            if any(v is not None for v in ys):
                plt.plot(xs, ys, color=color, linestyle=ls,
                         label='%s %s' % (name, key), linewidth=1.2)
    plt.xlabel('Optimizer Updates')
    plt.ylabel('alpha / beta')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out / 'alpha_beta_curve.png', dpi=140)
    plt.close()

    # delta PSNR distribution if paired metrics exist
    paired_path = out / 'paired_val_metrics.json'
    if not paired_path.exists():
        paired_path = Path(a.out_dir) / 'paired_val_metrics.json'
    # also check compare dir sibling
    candidates = [
        out / 'paired_val_metrics.json',
        Path(a.a2_dir).parent / 'compare' / 'paired_val_metrics.json',
    ]
    for p in candidates:
        if p.exists():
            data = json.load(open(p))
            deltas = [x['Delta_PSNR'] for x in data.get('per_image', [])]
            if deltas:
                plt.figure(figsize=(7, 4))
                plt.hist(deltas, bins=20, color='#555555', edgecolor='white')
                plt.axvline(np.mean(deltas), color='#d62728', label='mean')
                plt.xlabel(r'$\Delta$PSNR (A3 - A2)')
                plt.ylabel('Count')
                plt.legend()
                plt.tight_layout()
                plt.savefig(out / 'delta_psnr_distribution.png', dpi=140)
                plt.close()
            break
    print('Wrote plots to', out)


if __name__ == '__main__':
    main()
