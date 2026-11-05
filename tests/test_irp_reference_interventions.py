"""Reference interventions change only the Ref tensor path."""
from __future__ import annotations

import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.data.irp_dataset import TripletIRPDataset  # noqa: E402
from asrt.data.reference_modes import build_wrong_mapping  # noqa: E402
from tests.irp_test_fixtures import make_synthetic_triplet_dataset  # noqa: E402


def test_interventions_keep_low_gt_change_ref():
    with tempfile.TemporaryDirectory() as td:
        synth = make_synthetic_triplet_dataset(td, n_train=4, n_val=0, size=96, seed=0)
        rows = synth['train_rows']
        wmap = build_wrong_mapping([r['sample_id'] for r in rows], seed=0)
        id_to = {str(r['sample_id']): r for r in rows}
        base = TripletIRPDataset(
            rows, crop=0, seed=0, augment=False, halo=0, require_ref=True,
            ref_mode='correct', id_to_row=id_to)
        b0 = base[0]
        for mode in ('wrong', 'patch_shuffle', 'pixel_shuffle', 'ref_low'):
            ds = TripletIRPDataset(
                rows, crop=0, seed=0, augment=False, halo=0, require_ref=True,
                ref_mode=mode, wrong_mapping=wmap, id_to_row=id_to)
            b = ds[0]
            assert b['sample_id'] == b0['sample_id']
            assert torch.equal(b['low'], b0['low'])
            assert torch.equal(b['gt'], b0['gt'])
            assert not torch.equal(b['ref'], b0['ref']), mode
