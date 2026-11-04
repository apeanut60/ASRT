"""Augmentation epoch increment and A2/A3 shared geometry."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.irp_dataset import TripletIRPDataset  # noqa: E402
from asrt.data.manifest import load_manifest  # noqa: E402


def _few_rows(n=4):
    path = '/root/data/experiments/ASRT/manifests/lolv2_real_triplet_train.json'
    if not os.path.exists(path):
        return None
    rows = load_manifest(path)[:n]
    ref_root = '/root/data/datasets/lol-v2-real/Train/nanobanana_ref_v2'
    for r in rows:
        if not r.get('ref_path'):
            cand = os.path.join(ref_root, str(r['sample_id']) + '.png')
            if os.path.exists(cand):
                r['ref_path'] = cand
            else:
                # try low basename
                import pathlib
                cand2 = os.path.join(ref_root, pathlib.Path(r['low_path']).name)
                if os.path.exists(cand2):
                    r['ref_path'] = cand2
    rows = [r for r in rows if r.get('ref_path') and os.path.exists(r['ref_path'])]
    return rows if len(rows) >= 2 else None


def test_epoch_changes_crop():
    rows = _few_rows()
    if rows is None:
        print('SKIP test_epoch_changes_crop (no data)')
        return
    ds = TripletIRPDataset(
        rows, crop=128, seed=42, augment=True, epoch=0, halo=0, require_ref=True)
    a0 = ds[0]
    ds.set_epoch(1)
    a1 = ds[0]
    # same sample id, different crop/aug expected with high probability
    assert a0['sample_id'] == a1['sample_id']
    assert not torch.allclose(a0['low'], a1['low']), 'epoch must change augmentation'


def test_same_seed_reproducible():
    rows = _few_rows()
    if rows is None:
        print('SKIP test_same_seed_reproducible (no data)')
        return
    d1 = TripletIRPDataset(rows, crop=128, seed=7, epoch=2, halo=0, require_ref=True)
    d2 = TripletIRPDataset(rows, crop=128, seed=7, epoch=2, halo=0, require_ref=True)
    x1, x2 = d1[1], d2[1]
    assert torch.allclose(x1['low'], x2['low'])
    assert torch.allclose(x1['gt'], x2['gt'])
    assert torch.allclose(x1['ref'], x2['ref'])


def test_a2_a3_same_aug_params():
    """A2/A3 datasets with same seed/epoch/idx share Low/GT geometry."""
    rows = _few_rows()
    if rows is None:
        print('SKIP test_a2_a3_same_aug_params (no data)')
        return
    # Geometry determined before ref_mode; both use correct loading then mode
    d_a3 = TripletIRPDataset(
        rows, crop=128, seed=42, epoch=0, halo=0, require_ref=True, ref_mode='correct')
    d_low = TripletIRPDataset(
        rows, crop=128, seed=42, epoch=0, halo=0, require_ref=True, ref_mode='ref_low')
    a, b = d_a3[0], d_low[0]
    assert torch.allclose(a['low'], b['low'])
    assert torch.allclose(a['gt'], b['gt'])
    # ref differs: correct nano vs low
    assert not torch.allclose(a['ref'], b['ref'])


def main():
    test_epoch_changes_crop()
    test_same_seed_reproducible()
    test_a2_a3_same_aug_params()
    print('ALL test_irp_data_reproducibility PASS')


if __name__ == '__main__':
    main()
