"""IRP training / validation datasets with halo crop and ref-mode support."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from asrt.data.halo_crop import compute_irp_halo, sync_geometry_halo
from asrt.data.paired import read_rgb_m11
from asrt.data.reference_modes import (
    apply_reference_mode,
    load_wrong_mapping,
    resolve_wrong_ref_id,
)


def manifest_hash(rows: List[dict]) -> str:
    payload = json.dumps(
        [dict(sample_id=r.get('sample_id'), low=r.get('low_path'),
              gt=r.get('gt_path'), ref=r.get('ref_path')) for r in rows],
        sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def assert_no_overlap(train_rows: List[dict], val_rows: List[dict]):
    tr = {str(r['sample_id']) for r in train_rows}
    va = {str(r['sample_id']) for r in val_rows}
    both = tr & va
    if both:
        raise RuntimeError(
            'train/val sample_id overlap (%d): e.g. %s'
            % (len(both), sorted(list(both))[:5]))


class TripletIRPDataset(Dataset):
    """Low/GT/Ref with optional halo crop. Returns [-1,1] tensors.

    When halo>0 and return_halo=True (training), Low/GT/Ref include halo; the
    trainer crops predictions/GT to the valid center for the loss.
    """

    def __init__(
        self,
        rows: List[dict],
        crop: int = 128,
        seed: int = 42,
        augment: bool = True,
        epoch: int = 0,
        halo: int = 0,
        window_size: int = 8,
        sigma: float = 7.0,
        return_halo: bool = True,
        require_ref: bool = False,
        ref_mode: str = 'correct',
        wrong_mapping: Optional[Dict[str, str]] = None,
        wrong_mapping_path: Optional[str] = None,
        id_to_row: Optional[Dict[str, dict]] = None,
        patch_shuffle_size: int = 32,
        align_valid_to_window: bool = True,
    ):
        self.rows = list(rows)
        self.crop = int(crop)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = int(epoch)
        if halo is None or int(halo) < 0:
            halo = compute_irp_halo(sigma=sigma, window_size=window_size)
        self.halo = int(halo)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.return_halo = bool(return_halo) and self.halo > 0 and self.augment
        self.require_ref = bool(require_ref)
        self.ref_mode = str(ref_mode)
        self.patch_shuffle_size = int(patch_shuffle_size)
        self.align_valid_to_window = bool(align_valid_to_window)
        self.wrong_mapping = wrong_mapping
        if wrong_mapping_path and self.wrong_mapping is None:
            self.wrong_mapping = load_wrong_mapping(wrong_mapping_path)
        self.id_to_row = id_to_row or {
            str(r['sample_id']): r for r in self.rows}
        if self.require_ref:
            missing = [r['sample_id'] for r in self.rows if not r.get('ref_path')]
            if missing:
                raise RuntimeError(
                    'A3 Correct-ref training requires ref_path; missing e.g. %s'
                    % missing[:5])

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def _load_triplet(self, row: dict):
        low = read_rgb_m11(row['low_path'])
        gt = read_rgb_m11(row['gt_path'])
        ref = None
        if row.get('ref_path'):
            ref = read_rgb_m11(row['ref_path'])
        if tuple(gt.shape[-2:]) != tuple(low.shape[-2:]):
            gt = torch.nn.functional.interpolate(
                gt[None], size=low.shape[-2:], mode='bilinear', align_corners=False)[0]
        if ref is not None and tuple(ref.shape[-2:]) != tuple(low.shape[-2:]):
            ref = torch.nn.functional.interpolate(
                ref[None], size=low.shape[-2:], mode='bilinear', align_corners=False)[0]
        return low, gt, ref

    def __getitem__(self, idx):
        row = self.rows[idx]
        sample_id = str(row.get('sample_id', idx))
        low, gt, ref = self._load_triplet(row)
        if self.require_ref and ref is None:
            raise RuntimeError('missing ref for sample %s' % sample_id)

        # Wrong-ref resolved from fixed mapping BEFORE geometry so the same
        # crop/aug is applied to Low/GT/CorrectRef and WrongRef.
        wrong_ref = None
        wrong_id = None
        if self.ref_mode == 'wrong':
            if not self.wrong_mapping:
                raise RuntimeError('ref_mode=wrong requires wrong_mapping')
            wrong_id = resolve_wrong_ref_id(sample_id, self.wrong_mapping)
            wrow = self.id_to_row.get(wrong_id)
            if wrow is None:
                raise KeyError('wrong sample %r not in id_to_row' % wrong_id)
            _, _, wrong_ref = self._load_triplet(wrow)
            if wrong_ref is None:
                raise RuntimeError('wrong sample %r has no ref_path' % wrong_id)

        rng = np.random.RandomState(self.seed + self.epoch * 1000003 + int(idx))
        meta = dict(halo=0, crop=self.crop, top=0, left=0, flip_w=False,
                    flip_h=False, rot_k=0)
        if self.augment and self.crop > 0:
            tensors = [low, gt]
            if ref is not None:
                tensors.append(ref)
            if wrong_ref is not None:
                tensors.append(wrong_ref)
            packed = sync_geometry_halo(
                tensors, self.crop, rng, halo=self.halo if self.return_halo else 0,
                window_size=self.window_size,
                align_valid_to_window=self.align_valid_to_window)
            meta = dict(packed['meta'])
            if self.return_halo:
                seq = packed['tensors_halo']
            else:
                seq = packed['tensors_valid']
            low, gt = seq[0], seq[1]
            cursor = 2
            if ref is not None:
                ref = seq[cursor]
                cursor += 1
            if wrong_ref is not None:
                wrong_ref = seq[cursor]

        # Apply ref mode on the (halo or valid) tensors
        ref_for_model = apply_reference_mode(
            low[None] if low.dim() == 3 else low,
            None if ref is None else (ref[None] if ref.dim() == 3 else ref),
            mode=self.ref_mode,
            wrong_ref=None if wrong_ref is None else (
                wrong_ref[None] if wrong_ref.dim() == 3 else wrong_ref),
            patch_size=self.patch_shuffle_size,
        )
        if ref_for_model is not None and ref_for_model.dim() == 4:
            ref_for_model = ref_for_model[0]

        return dict(
            low=low,
            gt=gt,
            ref=ref_for_model,
            correct_ref=ref,
            sample_id=sample_id,
            has_ref=ref_for_model is not None,
            wrong_id=wrong_id,
            halo=int(meta.get('halo', 0)),
            crop=int(meta.get('crop', self.crop)),
            aug_meta={k: meta[k] for k in (
                'top', 'left', 'flip_w', 'flip_h', 'rot_k', 'halo', 'crop')
                if k in meta},
            ref_mode=self.ref_mode,
            domain=row.get('domain', ''),
        )


def collate_irp(batch: List[dict]) -> dict:
    """Stack IRP batch; allow ref=None (NoRef) as a None field."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ('low', 'gt', 'correct_ref'):
            if vals[0] is None:
                out[k] = None
            else:
                out[k] = torch.stack(vals, 0)
        elif k == 'ref':
            if any(v is None for v in vals):
                # mixed None not supported in a single batch for training
                if all(v is None for v in vals):
                    out[k] = None
                else:
                    raise RuntimeError(
                        'mixed ref/None in one batch; use homogeneous ref_mode')
            else:
                out[k] = torch.stack(vals, 0)
        elif k in ('halo', 'crop'):
            out[k] = int(vals[0])
        else:
            out[k] = vals
    return out
