"""Strict paired / triplet manifests. Never substitute GT for missing Ref."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _list_images(directory: str) -> Dict[str, str]:
    out = {}
    if not os.path.isdir(directory):
        return out
    for name in os.listdir(directory):
        if name.startswith('.'):
            continue
        low = name.lower()
        if not low.endswith(IMG_EXTS):
            continue
        stem = os.path.splitext(name)[0]
        out[stem] = os.path.join(directory, name)
    return out


def _stem_map(low_name: str) -> str:
    """Map low*.png stem to normal*.png stem for LOLv2."""
    if low_name.startswith('low'):
        return 'normal' + low_name[3:]
    return low_name


def build_paired(low_dir: str, gt_dir: str) -> List[dict]:
    lows = _list_images(low_dir)
    gts = _list_images(gt_dir)
    rows = []
    missing_gt = []
    for stem, low_path in sorted(lows.items()):
        gt_stem = _stem_map(stem)
        gt_path = gts.get(gt_stem) or gts.get(stem)
        if gt_path is None:
            missing_gt.append(stem)
            continue
        rows.append(dict(
            sample_id=stem, low_path=low_path, gt_path=gt_path, ref_path=None))
    if missing_gt:
        raise SystemExit(
            'paired: %d Low files missing GT (e.g. %s)' % (
                len(missing_gt), missing_gt[:5]))
    return rows


def build_triplet(low_dir: str, gt_dir: str, ref_dir: str,
                  require_ref: bool = True) -> List[dict]:
    lows = _list_images(low_dir)
    gts = _list_images(gt_dir)
    refs = _list_images(ref_dir)
    rows, missing_gt, missing_ref = [], [], []
    for stem, low_path in sorted(lows.items()):
        gt_stem = _stem_map(stem)
        gt_path = gts.get(gt_stem) or gts.get(stem)
        if gt_path is None:
            missing_gt.append(stem)
            continue
        # Ref files for LOLv2 nano are named like Low (lowXXXX).
        ref_path = refs.get(stem) or refs.get(gt_stem)
        if ref_path is None:
            missing_ref.append(stem)
            if require_ref:
                continue
            rows.append(dict(
                sample_id=stem, low_path=low_path, gt_path=gt_path, ref_path=None))
            continue
        rows.append(dict(
            sample_id=stem, low_path=low_path, gt_path=gt_path, ref_path=ref_path))
    if missing_gt:
        raise SystemExit(
            'triplet: %d Low missing GT (e.g. %s)' % (len(missing_gt), missing_gt[:5]))
    if require_ref and missing_ref:
        # Strict: do not shrink silently — report and keep only complete triplets.
        pass
    return rows, missing_ref


def build_manifest(name: str, low_dir: str, gt_dir: str,
                   ref_dir: Optional[str] = None, require_ref: bool = True) -> dict:
    if ref_dir:
        rows, missing_ref = build_triplet(low_dir, gt_dir, ref_dir, require_ref=require_ref)
        kind = 'triplet'
    else:
        rows = build_paired(low_dir, gt_dir)
        missing_ref = []
        kind = 'paired'
    blob = json.dumps([r['sample_id'] for r in rows], sort_keys=True).encode()
    return dict(
        name=name,
        kind=kind,
        n=len(rows),
        low_dir=low_dir,
        gt_dir=gt_dir,
        ref_dir=ref_dir,
        missing_ref=len(missing_ref),
        missing_ref_examples=missing_ref[:10],
        manifest_sha256=hashlib.sha256(blob).hexdigest(),
        rows=rows,
    )


def audit_dataset(roots: dict) -> dict:
    """roots: mapping of split_name -> {low, gt, ref?}."""
    report = {}
    for name, paths in roots.items():
        report[name] = build_manifest(
            name,
            paths['low'],
            paths['gt'],
            paths.get('ref'),
            require_ref=bool(paths.get('require_ref', True)),
        )
        # Drop heavy rows from summary copy
        summary = {k: v for k, v in report[name].items() if k != 'rows'}
        report[name + '_summary'] = summary
    return report


def split_dev(rows: List[dict], frac: float = 0.1, seed: int = 42) -> tuple:
    import numpy as np
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(rows))
    n_dev = max(1, int(round(len(rows) * frac))) if len(rows) else 0
    if len(rows) < 2:
        return list(rows), []
    n_dev = min(n_dev, len(rows) - 1)
    dev_i = set(int(i) for i in idx[:n_dev])
    train, dev = [], []
    for i, r in enumerate(rows):
        (dev if i in dev_i else train).append(r)
    return train, dev


def save_manifest(path: str, rows: List[dict], meta: Optional[dict] = None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = dict(meta or {}, n=len(rows), rows=rows)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_manifest(path: str) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    return data['rows'] if isinstance(data, dict) and 'rows' in data else data
