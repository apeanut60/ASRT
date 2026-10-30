#!/usr/bin/env python
"""Train ASRT from a Stage YAML config."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.manifest import load_manifest  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402
from asrt.training.trainer import Trainer  # noqa: E402


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_rows(cfg):
    stage = str(cfg.get('stage', 'a')).lower()
    data = cfg.get('data', {})
    man_root = data.get('manifest_root', '/root/data/experiments/ASRT/manifests')

    def load(name):
        path = os.path.join(man_root, name)
        if not os.path.isfile(path):
            raise SystemExit('missing manifest %s — run scripts/audit_data.py' % path)
        return load_manifest(path)

    if stage in ('a', 'stage_a'):
        noref = load('lolv2_real_paired_train.json') + load('lolv2_syn_paired_train.json')
        for r in noref:
            if 'syn' in r.get('domain', '') or 'synthetic' in r.get('low_path', ''):
                r['domain'] = 'syn'
            else:
                r['domain'] = 'real'
        # For Stage A Syn crop differs — Trainer uses crop_noref; mix shares crop.
        # Prefer real crop when mixed; smoke OK. Full training may split loaders later.
        eval_rows = load('lolv2_real_paired_dev.json')
        return dict(train_rows_noref=noref, train_rows_ref=None, eval_rows=eval_rows)

    if stage in ('b', 'stage_b', 'b_v4', 'stage_b_v4', 'v4'):
        domain = str(data.get('domain', 'all')).lower()
        if domain in ('syn', 'synthetic'):
            ref = load('lolv2_syn_triplet_train.json')
            noref = load('lolv2_syn_paired_train.json')
            try:
                eval_rows = load('lolv2_syn_triplet_dev.json')
            except SystemExit:
                eval_rows = load('lolv2_syn_paired_dev.json')
        elif domain in ('mit', 'fivek', 'mit_test'):
            ref = load('mit_triplet_train.json')
            noref = load('mit_paired_train.json')
            try:
                eval_rows = load('mit_triplet_dev.json')
            except SystemExit:
                eval_rows = load('mit_paired_dev.json')
        elif domain in ('real',):
            ref = load('lolv2_real_triplet_train.json')
            noref = load('lolv2_real_paired_train.json')
            try:
                eval_rows = load('lolv2_real_triplet_dev.json')
            except SystemExit:
                eval_rows = load('lolv2_real_paired_dev.json')
        else:
            ref = load('lolv2_real_triplet_train.json') + load('lolv2_syn_triplet_train.json')
            noref = load('lolv2_real_paired_train.json') + load('lolv2_syn_paired_train.json')
            # Prefer ref-capable unified Dev; fallback to paired_dev name if present.
            try:
                eval_rows = load('lolv2_real_triplet_dev.json')
            except SystemExit:
                eval_rows = load('lolv2_real_paired_dev.json')
        for r in ref + noref + eval_rows:
            if str(r.get('domain', '')).lower() == 'mit' or '/MIT/' in str(
                    r.get('low_path', '')):
                r['domain'] = 'mit'
            elif 'syn' in str(r.get('domain', '')).lower() or 'synthetic' in str(
                    r.get('low_path', '')).lower():
                r['domain'] = 'syn'
            elif not r.get('domain'):
                r['domain'] = 'real'
        return dict(train_rows_ref=ref, train_rows_noref=noref, eval_rows=eval_rows)

    # Stage C
    man = data.get('manifest')
    if not man or not os.path.isfile(man):
        # fallback to audited name
        cam = str(data.get('camera', 'Huawei')).lower()
        man = os.path.join(man_root, '%s_triplet_train.json' % cam)
    rows = load_manifest(man) if man.endswith('.json') else load(os.path.basename(man))
    # if full file without split
    if not man.endswith('_train.json'):
        from asrt.data.manifest import split_dev
        train, dev = split_dev(rows, frac=0.1, seed=int(cfg.get('seed', 42)))
    else:
        train = rows
        cam = str(data.get('camera', 'Huawei')).lower()
        dev_path = os.path.join(man_root, '%s_triplet_dev.json' % cam)
        dev = load_manifest(dev_path) if os.path.isfile(dev_path) else train[:20]
    return dict(train_rows_ref=train, train_rows_noref=None, eval_rows=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--max_steps', type=int, default=0, help='override for smoke')
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--resume', default='')
    ap.add_argument('--pretrained', default='')
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.max_steps:
        cfg['max_steps'] = a.max_steps
        cfg['total_steps'] = a.max_steps
    if a.out_dir:
        cfg['out_dir'] = a.out_dir
    # Seed before model/dataloader construction (Trainer also re-seeds)
    seed_everything(int(cfg.get('seed', 42)))
    rows = resolve_rows(cfg)
    trainer = Trainer(cfg)
    pre = a.pretrained or cfg.get('pretrained') or ''
    if a.resume:
        trainer.load(a.resume, resume_optim=True)
    elif pre and os.path.isfile(pre):
        trainer.load(pre, resume_optim=False)
    trainer.train(**rows)


if __name__ == '__main__':
    main()
