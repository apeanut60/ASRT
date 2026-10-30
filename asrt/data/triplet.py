"""Low/GT/Ref triplet dataset. Missing ref raises — never fill with GT."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from asrt.data.paired import read_rgb_m11, sync_geometry
from asrt.models.query_illum_align import estimate_query_align_maps


class TripletRefDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        crop: int = 256,
        seed: int = 42,
        augment: bool = True,
        epoch: int = 0,
        allow_missing_ref: bool = False,
        query_align: Optional[dict] = None,
    ):
        self.rows = list(rows)
        self.crop = int(crop)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = int(epoch)
        self.allow_missing_ref = bool(allow_missing_ref)
        self.query_align = dict(query_align or {})
        if not allow_missing_ref:
            bad = [r['sample_id'] for r in self.rows if not r.get('ref_path')]
            if bad:
                raise SystemExit(
                    'TripletRefDataset: %d rows missing ref_path (e.g. %s). '
                    'Do not substitute GT.' % (len(bad), bad[:5]))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def _estimate_full_align(self, low, ref):
        """Estimate gamma/LUT on full-resolution tensors before crop."""
        cfg = self.query_align
        if not cfg.get('enabled', False):
            return None, None
        alpha = float(cfg.get('gamma_alpha', 0.75))
        method = str(cfg.get('method', 'gamma_hist')).lower()
        if method in ('none', 'off', 'raw'):
            return None, None
        if method == 'gamma':
            # still estimate lut for API uniformity but beta=0 later
            pass
        gamma, lut, _ = estimate_query_align_maps(
            low[None], ref[None],
            alpha=alpha,
            gamma_min=float(cfg.get('gamma_min', 0.3)),
            gamma_max=float(cfg.get('gamma_max', 1.0)),
            bins=int(cfg.get('hist_bins', 256)),
        )
        return gamma[0].detach().cpu(), lut[0].detach().cpu()

    def __getitem__(self, idx):
        row = self.rows[idx]
        low = read_rgb_m11(row['low_path'])
        gt = read_rgb_m11(row['gt_path'])
        ref_path = row.get('ref_path')
        if ref_path:
            ref = read_rgb_m11(ref_path)
            has_ref = True
        else:
            if not self.allow_missing_ref:
                raise RuntimeError('missing ref for %s' % row['sample_id'])
            ref = None
            has_ref = False
        hw = low.shape[-2:]
        if tuple(gt.shape[-2:]) != hw:
            gt = torch.nn.functional.interpolate(
                gt[None], size=hw, mode='bilinear', align_corners=False)[0]
        if ref is not None and tuple(ref.shape[-2:]) != hw:
            ref = torch.nn.functional.interpolate(
                ref[None], size=hw, mode='bilinear', align_corners=False)[0]

        # Full-image alignment stats BEFORE geometry crop (stats_scope=full_image).
        query_gamma, query_lut = (None, None)
        if ref is not None:
            query_gamma, query_lut = self._estimate_full_align(low, ref)

        rng = np.random.RandomState(self.seed + self.epoch * 1000003 + idx)
        if self.augment and self.crop > 0:
            if ref is None:
                low, gt = sync_geometry([low, gt], self.crop, rng)
            else:
                low, gt, ref = sync_geometry([low, gt, ref], self.crop, rng)
        return dict(
            low=low, gt=gt, ref=ref,
            sample_id=row['sample_id'], has_ref=has_ref,
            domain=row.get('domain', ''),
            query_gamma=query_gamma,
            query_lut=query_lut,
        )


def collate_skip_none_ref(batch):
    """Stack batch; refs may be None for no-ref mixtures — keep list then."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if k == 'ref':
            if all(v is None for v in vals):
                out[k] = None
            elif any(v is None for v in vals):
                raise RuntimeError(
                    'mixed has_ref in one batch; sample no-ref separately')
            else:
                out[k] = torch.stack(vals, 0)
        elif k in ('query_gamma', 'query_lut'):
            if all(v is None for v in vals):
                out[k] = None
            elif any(v is None for v in vals):
                raise RuntimeError('mixed query_align tensors in one batch')
            else:
                out[k] = torch.stack(vals, 0)
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, 0)
        else:
            out[k] = vals
    return out
