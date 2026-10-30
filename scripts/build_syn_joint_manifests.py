#!/usr/bin/env python
"""Build Syn-only joint-train manifests with isolated Ref Dev (~30).

Canonical Dev = random sample of Nano-capable Syn Train IDs (seed=42).
NoRef Train / Ref Train exclude ALL Dev IDs; disjoint from official Syn Test.
Does not modify manifests/ or manifests_b_v2/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.manifest import load_manifest, save_manifest  # noqa: E402


def _ids(rows):
    return set(r['sample_id'] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/root/data/experiments/ASRT/manifests')
    ap.add_argument('--out', default='/root/data/experiments/ASRT/manifests_syn_joint')
    ap.add_argument('--dev_n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--audit', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    audit_path = a.audit or os.path.join(ROOT, 'docs', 'SYN_JOINT_SPLIT_AUDIT.md')

    syn_paired_all = load_manifest(os.path.join(a.src, 'lolv2_syn_paired.json'))
    syn_tri_all = load_manifest(os.path.join(a.src, 'lolv2_syn_triplet.json'))
    # Prefer full paired; fallback to train+dev concat if only splits exist
    if not syn_paired_all:
        syn_paired_all = (
            load_manifest(os.path.join(a.src, 'lolv2_syn_paired_train.json'))
            + load_manifest(os.path.join(a.src, 'lolv2_syn_paired_dev.json')))
    if not syn_tri_all:
        syn_tri_all = (
            load_manifest(os.path.join(a.src, 'lolv2_syn_triplet_train.json'))
            + load_manifest(os.path.join(a.src, 'lolv2_syn_triplet_dev.json')))

    test_path = os.path.join(a.src, 'lolv2_syn_test_triplet.json')
    if not os.path.isfile(test_path):
        raise SystemExit('missing %s — build Syn Test manifest first' % test_path)
    syn_test = load_manifest(test_path)
    test_ids = _ids(syn_test)

    # Nano-capable pool for Dev/Ref train
    ref_pool = [
        dict(r, domain='syn') for r in syn_tri_all
        if r.get('ref_path') and r['sample_id'] not in test_ids
    ]
    if len(ref_pool) < a.dev_n + 1:
        raise SystemExit('not enough Nano-capable Syn samples: %d' % len(ref_pool))

    rng = np.random.RandomState(a.seed)
    order = rng.permutation(len(ref_pool))
    n_dev = min(a.dev_n, len(ref_pool) - 1)
    dev_idx = set(int(i) for i in order[:n_dev])
    ref_dev = [ref_pool[i] for i in range(len(ref_pool)) if i in dev_idx]
    ref_train = [ref_pool[i] for i in range(len(ref_pool)) if i not in dev_idx]
    dev_ids = _ids(ref_dev)

    # NoRef Dev = same IDs from paired (drop ref)
    paired_map = {r['sample_id']: r for r in syn_paired_all}
    noref_dev = []
    for sid in sorted(dev_ids):
        if sid not in paired_map:
            raise SystemExit('Dev ID %s missing from syn paired' % sid)
        noref_dev.append(dict(paired_map[sid], domain='syn', ref_path=None))

    noref_train = [
        dict(r, domain='syn', ref_path=None) for r in syn_paired_all
        if r['sample_id'] not in dev_ids and r['sample_id'] not in test_ids
    ]

    # Sanity: every ref train/dev has Nano
    for r in ref_dev + ref_train:
        if not r.get('ref_path'):
            raise SystemExit('missing ref_path on %s' % r['sample_id'])

    sets = {
        'lolv2_syn_triplet_dev': ref_dev,
        'lolv2_syn_paired_dev': noref_dev,
        'lolv2_syn_triplet_train': ref_train,
        'lolv2_syn_paired_train': noref_train,
    }
    # Alias names expected by train.py stage-b loader when domain=syn
    for name, rows in sets.items():
        save_manifest(
            os.path.join(a.out, name + '.json'), rows,
            meta=dict(name=name, domain='syn', seed=a.seed, n=len(rows)))

    def inter(a, b):
        return _ids(a) & _ids(b)

    checks = {
        'Dev∩RefTrain': inter(ref_dev, ref_train),
        'Dev∩NoRefTrain': inter(ref_dev, noref_train),
        'Dev∩OfficialSynTest': inter(ref_dev, syn_test),
        'RefTrain∩OfficialSynTest': inter(ref_train, syn_test),
        'NoRefTrain∩OfficialSynTest': inter(noref_train, syn_test),
    }
    fail = [k for k, v in checks.items() if v]
    if fail:
        raise SystemExit('intersection FAIL: %s' % fail)

    lines = [
        '# Syn Joint Split Audit',
        '',
        'Generated: %s' % datetime.now(timezone.utc).isoformat(),
        '',
        '## Canonical Dev',
        '',
        '- Source: Nano-capable Syn Train sample (seed=%d, n=%d)' % (a.seed, n_dev),
        '- Ref Dev: **n=%d**' % len(ref_dev),
        '- NoRef Dev (same IDs): **n=%d**' % len(noref_dev),
        '',
        '## Train sizes',
        '',
        '| Split | n |',
        '|-------|---|',
        '| Syn Ref Train | %d |' % len(ref_train),
        '| Syn NoRef Train | %d |' % len(noref_train),
        '| Official Syn Test | %d |' % len(syn_test),
        '',
        '## Intersection checks (must be empty)',
        '',
    ]
    for k, v in checks.items():
        lines.append('- **%s**: PASS (n=0)' % k)
    lines += [
        '',
        '## Notes',
        '',
        '- Old manifests under `manifests/` / `manifests_b_v2/` are **not** overwritten.',
        '- New root: `%s`' % a.out,
        '- Ref Dev IDs sha256: `%s`' % hashlib.sha256(
            json.dumps(sorted(dev_ids)).encode()).hexdigest()[:16],
        '',
    ]
    os.makedirs(os.path.dirname(audit_path) or '.', exist_ok=True)
    with open(audit_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    # also copy audit next to manifests
    with open(os.path.join(a.out, 'SYN_JOINT_SPLIT_AUDIT.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('out=', a.out)
    print('ref_dev', len(ref_dev), 'ref_train', len(ref_train),
          'noref_train', len(noref_train))
    print('audit', audit_path)
    if fail:
        raise SystemExit(1)
    print('ALL_INTERSECTIONS_PASS')


if __name__ == '__main__':
    main()
