#!/usr/bin/env python
"""Warm-start ASRT backbone from legacy TTSR MainNet weights (safe subset only).

Never copies absolute RGB tails or old illum heads into the residual light-up model.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.asrt import ASRT  # noqa: E402


SKIP_SUBSTR = (
    'merge_tail', 'global_illum', 'ref_illum', 'conv_tail2',
    'illumination_estimator', 'coarse_igab', 'middle_igab',
    'conv11_head', 'conv22_head', 'conv33_head',
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--num_res_blocks', default='8+8+4+2')
    ap.add_argument('--n_feats', type=int, default=64)
    a = ap.parse_args()
    model = ASRT(num_res_blocks=a.num_res_blocks, n_feats=a.n_feats)
    dst = model.state_dict()
    src_raw = torch.load(a.input, map_location='cpu')
    if isinstance(src_raw, dict):
        for k in ('model', 'state_dict', 'MainNet', 'p2', 'net_g'):
            if k in src_raw and isinstance(src_raw[k], dict):
                src_raw = src_raw[k]
                break
    # strip prefixes
    src = {}
    for k, v in src_raw.items():
        nk = k
        for p in ('module.', 'MainNet.', 'backbone.', 'model.'):
            if nk.startswith(p):
                nk = nk[len(p):]
        src[nk] = v

    matched, skipped, missing = [], [], []
    new_sd = dict(dst)
    for k, v in dst.items():
        # map backbone.* from legacy MainNet keys without prefix
        key = k[len('backbone.'):] if k.startswith('backbone.') else None
        if key is None:
            continue
        if any(s in key for s in SKIP_SUBSTR):
            skipped.append(k)
            continue
        if key in src and src[key].shape == v.shape:
            new_sd[k] = src[key]
            matched.append(k)
        else:
            missing.append(k)
    model.load_state_dict(new_sd)
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    torch.save(dict(
        model=model.state_dict(),
        matched=len(matched), skipped=len(skipped),
        note='ASRT warmstart; RGB tail/illum/inject/IGAB randomly init'), a.output)
    print('matched=%d skipped=%d wrote %s' % (len(matched), len(skipped), a.output))


if __name__ == '__main__':
    main()
