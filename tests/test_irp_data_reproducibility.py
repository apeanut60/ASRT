"""Augmentation epoch increment and A2/A3 shared geometry (synthetic data)."""
from __future__ import annotations

import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.irp_dataset import TripletIRPDataset  # noqa: E402
from tests.irp_test_fixtures import make_synthetic_triplet_dataset  # noqa: E402


def _synth_rows(td: str, n: int = 4):
    synth = make_synthetic_triplet_dataset(td, n_train=n, n_val=2, size=192, seed=0)
    return synth['train_rows']


def test_epoch_changes_crop():
    with tempfile.TemporaryDirectory() as td:
        rows = _synth_rows(td)
        ds = TripletIRPDataset(
            rows, crop=64, seed=42, augment=True, epoch=0, halo=32, require_ref=True)
        a0 = ds[0]
        ds.set_epoch(1)
        a1 = ds[0]
        assert a0['sample_id'] == a1['sample_id']
        assert not torch.allclose(a0['low'], a1['low']), 'epoch must change augmentation'


def test_same_seed_reproducible():
    with tempfile.TemporaryDirectory() as td:
        rows = _synth_rows(td)
        d1 = TripletIRPDataset(rows, crop=64, seed=7, epoch=2, halo=32, require_ref=True)
        d2 = TripletIRPDataset(rows, crop=64, seed=7, epoch=2, halo=32, require_ref=True)
        x1, x2 = d1[1], d2[1]
        assert torch.allclose(x1['low'], x2['low'])
        assert torch.allclose(x1['gt'], x2['gt'])
        assert torch.allclose(x1['ref'], x2['ref'])


def test_a2_a3_same_aug_params():
    """A2/A3 datasets with same seed/epoch/idx share Low/GT geometry."""
    with tempfile.TemporaryDirectory() as td:
        rows = _synth_rows(td)
        d_a3 = TripletIRPDataset(
            rows, crop=64, seed=42, epoch=0, halo=32, require_ref=True, ref_mode='correct')
        d_low = TripletIRPDataset(
            rows, crop=64, seed=42, epoch=0, halo=32, require_ref=True, ref_mode='ref_low')
        a, b = d_a3[0], d_low[0]
        assert torch.allclose(a['low'], b['low'])
        assert torch.allclose(a['gt'], b['gt'])
        assert not torch.allclose(a['ref'], b['ref'])


def main():
    test_epoch_changes_crop()
    test_same_seed_reproducible()
    test_a2_a3_same_aug_params()
    print('ALL test_irp_data_reproducibility PASS')


if __name__ == '__main__':
    main()
