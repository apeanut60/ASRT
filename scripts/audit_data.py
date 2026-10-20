#!/usr/bin/env python
"""Scan LOLv2 / optional Huawei-Nikon Nano and write manifests."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.manifest import build_manifest, save_manifest, split_dev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/data/experiments/ASRT/manifests')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    specs = {
        'lolv2_real_paired': dict(
            low='/root/data/datasets/lol-v2-real/Train/Low',
            gt='/root/data/datasets/lol-v2-real/Train/Normal'),
        'lolv2_real_triplet': dict(
            low='/root/data/datasets/lol-v2-real/Train/Low',
            gt='/root/data/datasets/lol-v2-real/Train/Normal',
            ref='/root/data/datasets/lol-v2-real/Train/nanobanana_ref_v2'),
        'lolv2_syn_paired': dict(
            low='/root/data/datasets/lol-v2-synthetic/Train/Low',
            gt='/root/data/datasets/lol-v2-synthetic/Train/Normal'),
        'lolv2_syn_triplet': dict(
            low='/root/data/datasets/lol-v2-synthetic/Train/Low',
            gt='/root/data/datasets/lol-v2-synthetic/Train/Normal',
            ref='/root/data/datasets/lol-v2-synthetic/Train/nanobanana_ref'),
        'lolv2_real_test_paired': dict(
            low='/root/data/datasets/lol-v2-real/Test/Low',
            gt='/root/data/datasets/lol-v2-real/Test/Normal'),
        'lolv2_real_test_triplet': dict(
            low='/root/data/datasets/lol-v2-real/Test/Low',
            gt='/root/data/datasets/lol-v2-real/Test/Normal',
            ref='/root/data/datasets/lol-v2-real/Test/nanobanana_ref_v3'),
    }
    # Optional Huawei/Nikon Nano (budget ~500) if present
    data1 = '/root/data/datasets/data1'
    for cam in ('Huawei', 'Nikon'):
        base = os.path.join(data1, 'Training data', cam)
        low = next((os.path.join(base, n) for n in ('low', 'Low')
                    if os.path.isdir(os.path.join(base, n))), '')
        gt = next((os.path.join(base, n) for n in ('high', 'GT', 'Normal')
                   if os.path.isdir(os.path.join(base, n))), '')
        ref = os.path.join(base, 'nanobanana_ref')
        if low and gt and os.path.isdir(ref):
            specs['%s_triplet' % cam.lower()] = dict(low=low, gt=gt, ref=ref)

    report = {}
    for name, p in specs.items():
        if not os.path.isdir(p['low']):
            print('SKIP missing', name, p['low'])
            continue
        man = build_manifest(name, p['low'], p['gt'], p.get('ref'), require_ref=True)
        # Annotate domain: real | syn | huawei | nikon | ...
        for r in man['rows']:
            if 'syn' in name:
                r['domain'] = 'syn'
            elif 'real' in name:
                r['domain'] = 'real'
            elif 'huawei' in name:
                r['domain'] = 'huawei'
            elif 'nikon' in name:
                r['domain'] = 'nikon'
            else:
                r['domain'] = name.split('_')[0]
        path = os.path.join(a.out, name + '.json')
        save_manifest(path, man['rows'], meta={
            k: man[k] for k in man if k != 'rows'})
        train, dev = split_dev(man['rows'], frac=0.05, seed=42)
        save_manifest(os.path.join(a.out, name + '_train.json'), train)
        save_manifest(os.path.join(a.out, name + '_dev.json'), dev)
        summary = {k: man[k] for k in man if k != 'rows'}
        summary['n_train'] = len(train)
        summary['n_dev'] = len(dev)
        report[name] = summary
        print('%s n=%d missing_ref=%d sha=%s' % (
            name, man['n'], man['missing_ref'], man['manifest_sha256'][:12]), flush=True)

    with open(os.path.join(a.out, 'audit_summary.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print('wrote', os.path.join(a.out, 'audit_summary.json'))


if __name__ == '__main__':
    main()
