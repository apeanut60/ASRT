#!/usr/bin/env python
"""Build MIT joint-train manifests from testing split (Nano-capable scenes).

Low  = INPUT_IMAGES/{id}_N1.5.JPG
GT   = expert_c_testing_set/{id}.jpg
Ref  = nanobanana_ref_mit_v1/{id}_N1.5.JPG

Scenes without Nano are dropped (never GT-filled).
Dev ≈ 50 held out from the Nano-capable pool (seed=42).
No official MIT test protocol — testing images are the train pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.manifest import save_manifest  # noqa: E402


def _find(dir_path: Path, stem: str) -> str | None:
    for ext in ('.JPG', '.jpg', '.JPEG', '.jpeg', '.png', '.PNG'):
        p = dir_path / (stem + ext)
        if p.is_file():
            return str(p)
    return None


def _ids(rows):
    return set(r['sample_id'] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mit_test', default='/root/data/datasets/MIT/testing')
    ap.add_argument('--out', default='/root/data/experiments/ASRT/manifests_mit_joint')
    ap.add_argument('--dev_n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--expert', default='expert_c_testing_set')
    ap.add_argument('--audit', default='')
    a = ap.parse_args()

    root = Path(a.mit_test)
    inp_dir = root / 'INPUT_IMAGES'
    gt_dir = root / a.expert
    ref_dir = root / 'nanobanana_ref_mit_v1'
    for d in (inp_dir, gt_dir, ref_dir):
        if not d.is_dir():
            raise SystemExit('missing dir %s' % d)

    # Scene IDs from GT (no exposure suffix)
    gt_stems = []
    for p in sorted(gt_dir.iterdir()):
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and p.is_file():
            gt_stems.append(p.stem)

    triplets = []
    missing_ref = []
    missing_low = []
    for sid in gt_stems:
        low = _find(inp_dir, sid + '_N1.5')
        gt = _find(gt_dir, sid)
        ref = _find(ref_dir, sid + '_N1.5')
        if low is None:
            missing_low.append(sid)
            continue
        if gt is None:
            continue
        if ref is None:
            missing_ref.append(sid)
            continue
        triplets.append(dict(
            sample_id=sid,
            low_path=low,
            gt_path=gt,
            ref_path=ref,
            domain='mit',
        ))

    if len(triplets) < a.dev_n + 1:
        raise SystemExit('too few triplets: %d (missing_ref=%d missing_low=%d)' % (
            len(triplets), len(missing_ref), len(missing_low)))

    rng = np.random.RandomState(a.seed)
    order = rng.permutation(len(triplets))
    n_dev = min(a.dev_n, len(triplets) - 1)
    dev_idx = set(int(i) for i in order[:n_dev])
    ref_dev = [triplets[i] for i in range(len(triplets)) if i in dev_idx]
    ref_train = [triplets[i] for i in range(len(triplets)) if i not in dev_idx]
    dev_ids = _ids(ref_dev)

    noref_dev = [dict(r, ref_path=None) for r in ref_dev]
    noref_train = [dict(r, ref_path=None) for r in ref_train]

    os.makedirs(a.out, exist_ok=True)
    sets = {
        'mit_triplet_dev': ref_dev,
        'mit_paired_dev': noref_dev,
        'mit_triplet_train': ref_train,
        'mit_paired_train': noref_train,
    }
    for name, rows in sets.items():
        save_manifest(
            os.path.join(a.out, name + '.json'), rows,
            meta=dict(
                name=name, domain='mit', seed=a.seed, expert=a.expert,
                low_variant='_N1.5', n=len(rows),
                source=str(root),
            ))

    checks = {
        'Dev∩RefTrain': _ids(ref_dev) & _ids(ref_train),
        'Dev∩NoRefTrain': _ids(ref_dev) & _ids(noref_train),
    }
    fail = [k for k, v in checks.items() if v]
    if fail:
        raise SystemExit('intersection FAIL: %s' % fail)

    lines = [
        '# MIT Joint Split Audit (testing-as-train)',
        '',
        'Generated: %s' % datetime.now(timezone.utc).isoformat(),
        '',
        '## Source',
        '',
        '- Pool: MIT **testing** only (`%s`)' % root,
        '- Low: `INPUT_IMAGES/{id}_N1.5`',
        '- GT: `%s/{id}`' % a.expert,
        '- Ref: `nanobanana_ref_mit_v1/{id}_N1.5`',
        '- Dropped missing Nano: **%d** (examples %s)' % (
            len(missing_ref), missing_ref[:8]),
        '- Dropped missing Low: **%d**' % len(missing_low),
        '',
        '## Splits',
        '',
        '| Split | n |',
        '|-------|---|',
        '| Ref Dev | %d |' % len(ref_dev),
        '| Ref Train | %d |' % len(ref_train),
        '| NoRef Train | %d |' % len(noref_train),
        '| Triplet pool | %d |' % len(triplets),
        '',
        '## Intersection checks',
        '',
    ]
    for k in checks:
        lines.append('- **%s**: PASS (n=0)' % k)
    lines += [
        '',
        '## Notes',
        '',
        '- **No official MIT test eval** — testing images are the training pool.',
        '- Manifest root: `%s`' % a.out,
        '- Dev IDs sha256: `%s`' % hashlib.sha256(
            json.dumps(sorted(dev_ids)).encode()).hexdigest()[:16],
        '',
    ]
    audit_path = a.audit or os.path.join(ROOT, 'docs', 'MIT_JOINT_SPLIT_AUDIT.md')
    os.makedirs(os.path.dirname(audit_path) or '.', exist_ok=True)
    text = '\n'.join(lines) + '\n'
    with open(audit_path, 'w') as f:
        f.write(text)
    with open(os.path.join(a.out, 'MIT_JOINT_SPLIT_AUDIT.md'), 'w') as f:
        f.write(text)

    print('triplets', len(triplets), 'ref_dev', len(ref_dev),
          'ref_train', len(ref_train), 'missing_ref', len(missing_ref))
    print('audit', audit_path)
    print('ALL_INTERSECTIONS_PASS')


if __name__ == '__main__':
    main()
