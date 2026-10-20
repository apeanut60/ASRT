"""Adam param groups for Stage A/B/C."""
from __future__ import annotations

import torch


def build_optimizer(model, cfg: dict):
    stage = str(cfg.get('stage', 'a')).lower()
    base_lr = float(cfg.get('lr', 2e-4))
    betas = tuple(cfg.get('betas', (0.9, 0.999)))
    wd = float(cfg.get('weight_decay', 0.0))

    if stage in ('a', 'stage_a'):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=base_lr, betas=betas, weight_decay=wd)

    # Stage B/C: grouped LRs
    backbone = model.backbone
    groups = []
    # trunk
    trunk = []
    for name, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in (
                'conv11_head', 'conv22_head', 'conv33_head',
                'illumination_estimator', 'coarse_igab', 'middle_igab',
                'merge_tail')):
            continue
        trunk.append(p)
    groups.append(dict(params=trunk, lr=float(cfg.get('lr_backbone', 1e-5)), name='backbone'))

    inject = list(backbone.conv11_head.parameters()) + list(
        backbone.conv22_head.parameters()) + list(backbone.conv33_head.parameters())
    groups.append(dict(
        params=[p for p in inject if p.requires_grad],
        lr=float(cfg.get('lr_inject', 1e-4)), name='inject'))

    igab = list(backbone.coarse_igab.parameters()) + list(backbone.middle_igab.parameters())
    groups.append(dict(
        params=[p for p in igab if p.requires_grad],
        lr=float(cfg.get('lr_igab', 2e-5)), name='igab'))

    illum = list(backbone.illumination_estimator.parameters()) + list(
        backbone.merge_tail.parameters())
    groups.append(dict(
        params=[p for p in illum if p.requires_grad],
        lr=float(cfg.get('lr_illum', 2e-5)), name='illum_tail'))

    # filter empties
    groups = [g for g in groups if len(g['params']) > 0]
    return torch.optim.Adam(groups, betas=betas, weight_decay=wd)
