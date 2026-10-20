"""Low/GT paired dataset for Stage A (ref=None)."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_rgb_m11(path: str) -> torch.Tensor:
    im = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
    t = torch.from_numpy(im).permute(2, 0, 1) / 127.5 - 1.0
    return t


def sync_geometry(tensors: List[torch.Tensor], crop: int, rng: np.random.RandomState):
    """Apply one shared random crop (+ optional flips) to all tensors."""
    _, h, w = tensors[0].shape
    if h < crop or w < crop:
        pad_h = max(0, crop - h)
        pad_w = max(0, crop - w)
        tensors = [
            torch.nn.functional.pad(t[None], (0, pad_w, 0, pad_h), mode='reflect')[0]
            for t in tensors]
        _, h, w = tensors[0].shape
    top = int(rng.randint(0, h - crop + 1))
    left = int(rng.randint(0, w - crop + 1))
    out = [t[:, top:top + crop, left:left + crop] for t in tensors]
    if rng.rand() < 0.5:
        out = [torch.flip(t, dims=[-1]) for t in out]
    if rng.rand() < 0.5:
        out = [torch.flip(t, dims=[-2]) for t in out]
    if rng.rand() < 0.5:
        k = int(rng.randint(1, 4))
        out = [torch.rot90(t, k, dims=[-2, -1]) for t in out]
    return out


class PairedRestoreDataset(Dataset):
    def __init__(self, rows: List[dict], crop: int = 256, seed: int = 42,
                 augment: bool = True, epoch: int = 0):
        self.rows = list(rows)
        self.crop = int(crop)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        low = read_rgb_m11(row['low_path'])
        gt = read_rgb_m11(row['gt_path'])
        if tuple(gt.shape[-2:]) != tuple(low.shape[-2:]):
            gt = torch.nn.functional.interpolate(
                gt[None], size=low.shape[-2:], mode='bilinear', align_corners=False)[0]
        rng = np.random.RandomState(self.seed + self.epoch * 1000003 + idx)
        if self.augment and self.crop > 0:
            low, gt = sync_geometry([low, gt], self.crop, rng)
        return dict(
            low=low, gt=gt, ref=None,
            sample_id=row['sample_id'], has_ref=False, domain=row.get('domain', ''))
