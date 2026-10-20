#!/usr/bin/env python3
"""Build a v3 full checkpoint from legacy TTSR-lowlight weights.

Copy only shape-compatible internal MainNet layers that still have similar
semantic roles; do not copy old absolute-RGB tail or post-correction heads into
v3's residual-on-light-up architecture. New tensors keep v3 initialization.
The output is a full-model state_dict usable by the existing loader.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'model'))
from MainNetCSFIUNet import MainNetCSFIUNet


def weights(raw):
    if not isinstance(raw, dict):
        raise TypeError('Checkpoint must be a dict of tensors or wrapper dict')
    for k in ('state_dict', 'model_state_dict', 'params', 'model'):
        if k in raw and isinstance(raw[k], dict):
            raw = raw[k]
            break
    return {k.removeprefix('module.'): v for k, v in raw.items()
            if isinstance(v, torch.Tensor)}


def build(legacy, blocks, channels, copy_injection_heads=False):
    main = MainNetCSFIUNet(num_res_blocks=blocks, n_feats=channels)
    init_sd = main.state_dict()
    old_sd = weights(legacy)
    skip = ('ex12.', 'ex123.', 'merge_tail.', 'global_illum_head.',
            'ref_illum.', 'illumination_estimator.', 'coarse_igab.',
            'middle_igab.', 'down1.', 'down2.', 'skip_e1.', 'skip_e0.')
    if not copy_injection_heads:
        skip += ('conv11_head.', 'conv22_head.', 'conv33_head.')
    copied = []
    for k, v in init_sd.items():
        from_key = 'MainNet.' + k
        if k.startswith(skip):
            continue
        if from_key in old_sd and old_sd[from_key].shape == v.shape:
            init_sd[k] = old_sd[from_key].clone()
            copied.append(k)

    # Retain LTE, SearchTransfer, RefCorrection parameters if present.
    full = {k: v for k, v in old_sd.items() if not k.startswith('MainNet.')}
    full.update({'MainNet.' + k: v for k, v in init_sd.items()})
    return full, copied


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--num_res_blocks', default='8+8+4+2')
    p.add_argument('--n_feats', type=int, default=64)
    p.add_argument('--copy_injection_heads', action='store_true',
                   help='Copy legacy T injection heads despite new spatial stats')
    p.add_argument('--trust_checkpoint', action='store_true',
                   help='Only use for checkpoints from your own trusted training')
    args = p.parse_args()
    try:
        raw = torch.load(args.input, map_location='cpu', weights_only=True)
    except Exception:
        if not args.trust_checkpoint:
            raise
        raw = torch.load(args.input, map_location='cpu', weights_only=False)
    sd, copied = build(raw, [int(x) for x in args.num_res_blocks.split('+')],
                       args.n_feats, args.copy_injection_heads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.output)
    print(f'Saved {args.output}; copied {len(copied)} shape-compatible MainNet tensors.')
    print('Kept legacy non-MainNet tensors. All new modules initialized for v3.')
    print('DO NOT interpret this as a fully trained v3 model.')
