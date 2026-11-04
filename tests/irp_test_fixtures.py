"""Deterministic synthetic Low/GT/Ref triplets for IRP trainer unit tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode='RGB').save(path)


def make_synthetic_triplet_dataset(
    root: str,
    *,
    n_train: int = 8,
    n_val: int = 4,
    size: int = 192,
    seed: int = 0,
) -> Dict[str, object]:
    """Create Low/GT/Ref PNGs + train/val manifests under `root`.

    Train/val sample IDs do not overlap. Images are large enough for crop+halo.
    Ref is distinct from GT (never copied from GT).
    """
    root_p = Path(root)
    low_d = root_p / 'Low'
    gt_d = root_p / 'GT'
    ref_d = root_p / 'Ref'
    for d in (low_d, gt_d, ref_d):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    train_rows: List[dict] = []
    val_rows: List[dict] = []

    def _mk(sid: str) -> dict:
        # Distinct patterns per branch so wrong-ref / prior modes are meaningful
        low = rng.randint(0, 80, (size, size, 3), dtype=np.uint8)
        gt = rng.randint(100, 220, (size, size, 3), dtype=np.uint8)
        ref = rng.randint(40, 200, (size, size, 3), dtype=np.uint8)
        # Ensure ref != gt
        ref = np.clip(ref.astype(np.int16) + 17, 0, 255).astype(np.uint8)
        lp = low_d / f'{sid}.png'
        gp = gt_d / f'{sid}.png'
        rp = ref_d / f'{sid}.png'
        _write_rgb(lp, low)
        _write_rgb(gp, gt)
        _write_rgb(rp, ref)
        return dict(
            sample_id=sid,
            low_path=str(lp),
            gt_path=str(gp),
            ref_path=str(rp),
            domain='synth',
        )

    for i in range(n_train):
        train_rows.append(_mk('train_%03d' % i))
    for i in range(n_val):
        val_rows.append(_mk('val_%03d' % i))

    train_man = root_p / 'train.json'
    val_man = root_p / 'val.json'
    with open(train_man, 'w') as f:
        json.dump(dict(n=len(train_rows), rows=train_rows), f)
    with open(val_man, 'w') as f:
        json.dump(dict(n=len(val_rows), rows=val_rows), f)

    return dict(
        root=str(root_p),
        train_manifest=str(train_man),
        val_manifest=str(val_man),
        train_rows=train_rows,
        val_rows=val_rows,
        size=size,
    )


def apply_synth_paths(cfg: dict, synth: dict) -> dict:
    """Point cfg data/manifest fields at a synthetic dataset."""
    cfg = dict(cfg)
    data = dict(cfg.get('data') or {})
    data['manifest_root'] = synth['root']
    data['train_manifest'] = synth['train_manifest']
    data['val_manifest'] = synth['val_manifest']
    data['real_low'] = os.path.join(synth['root'], 'Low')
    data['real_gt'] = os.path.join(synth['root'], 'GT')
    data['real_ref'] = os.path.join(synth['root'], 'Ref')
    data['ref_mode'] = data.get('ref_mode', 'correct')
    cfg['data'] = data
    return cfg
