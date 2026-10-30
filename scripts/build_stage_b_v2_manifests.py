#!/usr/bin/env python
"""Build Stage B-v2 manifests with unified Dev isolation (no train leakage).

Canonical Dev = Stage A lolv2_real_paired_dev (verified disjoint from A train).
Ref Dev = those IDs that have a valid Nano Ref (never GT substitute).
All Stage B train loaders exclude ALL canonical Dev IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.manifest import load_manifest, save_manifest  # noqa: E402


def _ids(rows):
    return set(r['sample_id'] for r in rows)


def _filter_out(rows, ban):
    return [r for r in rows if r['sample_id'] not in ban]


def _sha_ids(rows):
    blob = json.dumps(sorted(_ids(rows)), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/root/data/experiments/ASRT/manifests')
    ap.add_argument('--out', default='/root/data/experiments/ASRT/manifests_b_v2')
    ap.add_argument('--audit', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    audit_path = a.audit or os.path.join(a.out, 'STAGE_B_V2_SPLIT_AUDIT.md')

    paired_dev = load_manifest(os.path.join(a.src, 'lolv2_real_paired_dev.json'))
    paired_train = load_manifest(os.path.join(a.src, 'lolv2_real_paired_train.json'))
    syn_paired_train = load_manifest(os.path.join(a.src, 'lolv2_syn_paired_train.json'))
    real_tri_all = load_manifest(os.path.join(a.src, 'lolv2_real_triplet.json'))
    syn_tri_all = load_manifest(os.path.join(a.src, 'lolv2_syn_triplet.json'))
    test_paired = load_manifest(os.path.join(a.src, 'lolv2_real_test_paired.json'))
    test_tri = load_manifest(os.path.join(a.src, 'lolv2_real_test_triplet.json'))

    # Also load old B splits for comparison reporting only
    old_tri_dev = load_manifest(os.path.join(a.src, 'lolv2_real_triplet_dev.json'))
    old_tri_train = load_manifest(os.path.join(a.src, 'lolv2_real_triplet_train.json'))

    dev_ids = _ids(paired_dev)
    test_ids = _ids(test_paired) | _ids(test_tri)

    # Validate Stage A Dev was never in A train
    a_train_ids = _ids(paired_train) | _ids(syn_paired_train)
    leak_a = dev_ids & a_train_ids
    if leak_a:
        raise SystemExit('FATAL: Stage A Dev overlaps A train: %s' % sorted(leak_a)[:10])

    # Ref-capable Dev subset (must have Nano; never GT)
    tri_map = {r['sample_id']: r for r in real_tri_all if r.get('ref_path')}
    ref_dev = []
    for r in paired_dev:
        sid = r['sample_id']
        if sid in tri_map:
            row = dict(tri_map[sid])
            row['domain'] = 'real'
            ref_dev.append(row)
    noref_dev = [dict(r, domain='real') for r in paired_dev]

    ban = set(dev_ids)  # exclude ALL canonical Dev IDs from every train loader

    real_ref_train = [
        dict(r, domain='real') for r in real_tri_all
        if r.get('ref_path') and r['sample_id'] not in ban and r['sample_id'] not in test_ids
    ]
    syn_ref_train = [
        dict(r, domain='syn') for r in syn_tri_all
        if r.get('ref_path') and r['sample_id'] not in ban and r['sample_id'] not in test_ids
    ]
    real_noref_train = [
        dict(r, domain='real', ref_path=None) for r in paired_train
        if r['sample_id'] not in ban and r['sample_id'] not in test_ids
    ]
    syn_noref_train = [
        dict(r, domain='syn', ref_path=None) for r in syn_paired_train
        if r['sample_id'] not in ban and r['sample_id'] not in test_ids
    ]

    sets = {
        'unified_real_dev_noref': noref_dev,
        'unified_real_dev_ref': ref_dev,
        'lolv2_real_triplet_train': real_ref_train,
        'lolv2_syn_triplet_train': syn_ref_train,
        'lolv2_real_paired_train': real_noref_train,
        'lolv2_syn_paired_train': syn_noref_train,
    }

    # Intersection checks
    checks = {}
    def chk(name, a, b):
        inter = _ids(a) & _ids(b)
        checks[name] = dict(n=len(inter), examples=sorted(inter)[:10])
        return inter

    fail = []
    if chk('Dev∩RealRefTrain', noref_dev, real_ref_train):
        fail.append('Dev∩RealRefTrain')
    if chk('Dev∩SynRefTrain', noref_dev, syn_ref_train):
        fail.append('Dev∩SynRefTrain')
    if chk('Dev∩RealNoRefTrain', noref_dev, real_noref_train):
        fail.append('Dev∩RealNoRefTrain')
    if chk('Dev∩SynNoRefTrain', noref_dev, syn_noref_train):
        fail.append('Dev∩SynNoRefTrain')
    if chk('Dev∩OfficialTest', noref_dev, test_paired):
        fail.append('Dev∩OfficialTest')
    if chk('RealRefTrain∩OfficialTest', real_ref_train, test_paired):
        fail.append('RealRefTrain∩OfficialTest')
    if chk('RealNoRefTrain∩OfficialTest', real_noref_train, test_paired):
        fail.append('RealNoRefTrain∩OfficialTest')
    if chk('SynRefTrain∩OfficialTest', syn_ref_train, test_tri if test_tri else test_paired):
        # syn shouldn't overlap real test by ID namespace usually
        pass
    chk('OldTriDev∩NewRealNoRefTrain(should_be_0_now)', old_tri_dev, real_noref_train)
    chk('A_Dev∩A_Train(must_0)', paired_dev, paired_train)

    # All ref_dev must have ref_path
    if any(not r.get('ref_path') for r in ref_dev):
        fail.append('ref_dev_missing_ref_path')
    if fail:
        raise SystemExit('SPLIT AUDIT FAILED: %s\n%s' % (fail, checks))

    meta_common = dict(
        version='b_v2',
        created_utc=datetime.now(timezone.utc).isoformat(),
        seed=42,
        canonical_dev='lolv2_real_paired_dev',
        ban_ids_sha256=hashlib.sha256(
            json.dumps(sorted(ban), sort_keys=True).encode()).hexdigest(),
    )
    for name, rows in sets.items():
        path = os.path.join(a.out, name + '.json')
        save_manifest(path, rows, meta={
            **meta_common,
            'name': name,
            'n': len(rows),
            'ids_sha256': _sha_ids(rows),
        })
        print('wrote', path, 'n=', len(rows), 'sha=', _sha_ids(rows)[:12])

    # Symlink-style convenience names for train.py Stage B
    # train.py loads lolv2_real_triplet_dev — write that as ref_dev
    save_manifest(
        os.path.join(a.out, 'lolv2_real_triplet_dev.json'), ref_dev,
        meta={**meta_common, 'name': 'lolv2_real_triplet_dev', 'n': len(ref_dev),
              'ids_sha256': _sha_ids(ref_dev)})
    save_manifest(
        os.path.join(a.out, 'lolv2_real_paired_dev.json'), noref_dev,
        meta={**meta_common, 'name': 'lolv2_real_paired_dev', 'n': len(noref_dev),
              'ids_sha256': _sha_ids(noref_dev)})

    lines = [
        '# Stage B-v2 Split Audit',
        '',
        'Generated: %s' % meta_common['created_utc'],
        '',
        '## Canonical Dev',
        '',
        '- Source: Stage A `lolv2_real_paired_dev` (**n=%d**)' % len(noref_dev),
        '- Stage A Dev ∩ Stage A Train: **%d** (must be 0)' % checks['A_Dev∩A_Train(must_0)']['n'],
        '- Ref-capable Dev (has Nano): **n=%d** (IDs without Nano excluded from ref eval, never GT-filled)' % len(ref_dev),
        '- Missing Nano among Dev: %s' % sorted(dev_ids - _ids(ref_dev)),
        '',
        '## Train sizes (after excluding ALL Dev IDs + Test IDs)',
        '',
        '| Split | n |',
        '|-------|---|',
        '| Real Ref Train | %d |' % len(real_ref_train),
        '| Syn Ref Train | %d |' % len(syn_ref_train),
        '| Real NoRef Train | %d |' % len(real_noref_train),
        '| Syn NoRef Train | %d |' % len(syn_noref_train),
        '',
        '## Intersection checks (must be empty)',
        '',
    ]
    for k, v in checks.items():
        status = 'PASS' if v['n'] == 0 else 'FAIL'
        lines.append('- **%s**: %s (n=%d) examples=%s' % (k, status, v['n'], v['examples']))
    lines += [
        '',
        '## Notes',
        '',
        '- Old Stage B `triplet_dev ∩ real_paired_train` leakage is **eliminated** in this version.',
        '- Old manifests under `/root/data/experiments/ASRT/manifests/` are **not overwritten**.',
        '- New root: `%s`' % a.out,
        '',
    ]
    open(audit_path, 'w').write('\n'.join(lines) + '\n')
    # also copy into docs
    docs = os.path.join(ROOT, 'docs', 'STAGE_B_V2_SPLIT_AUDIT.md')
    open(docs, 'w').write('\n'.join(lines) + '\n')
    print('audit ->', audit_path)
    print('audit ->', docs)
    if fail:
        raise SystemExit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
