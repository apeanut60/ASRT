"""Adam param groups for Stage A/B/C / v3."""
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

    # Stage B/C/v3: grouped LRs
    backbone = model.backbone
    assigned = set()

    def take(params):
        out = []
        for p in params:
            if not p.requires_grad:
                continue
            if id(p) in assigned:
                continue
            assigned.add(id(p))
            out.append(p)
        return out

    groups = []

    # --- new v3 modules (higher LR) ---
    v3_params = []
    if getattr(model, 'enable_v3', False):
        if model.reliability is not None:
            v3_params += list(model.reliability.parameters())
        for mod_name in ('coarse_illum_adapter', 'middle_illum_adapter', 'fine_ct_refine'):
            mod = getattr(backbone, mod_name, None)
            if mod is not None:
                v3_params += list(mod.parameters())
        # CondResBlock adapters only (keep conv1/conv2 in backbone/igab groups)
        for rb_list_name in ('RB11', 'RB22'):
            rbs = getattr(backbone, rb_list_name, None)
            if rbs is None:
                continue
            for rb in rbs:
                if hasattr(rb, 'adapter'):
                    v3_params += list(rb.adapter.parameters())
        v3_params = take(v3_params)
        if v3_params:
            groups.append(dict(
                params=v3_params,
                lr=float(cfg.get('lr_v3', 1e-4)),
                name='v3_new'))

    # trunk (exclude inject / illum / igab / merge / v3 adapters already taken)
    trunk = []
    for name, p in backbone.named_parameters():
        if not p.requires_grad or id(p) in assigned:
            continue
        if any(k in name for k in (
                'conv11_head', 'conv22_head', 'conv33_head',
                'illumination_estimator', 'coarse_igab', 'middle_igab',
                'merge_tail', 'coarse_illum_adapter', 'middle_illum_adapter',
                'fine_ct_refine')):
            continue
        # CondResBlock.adapter already taken
        if '.adapter.' in name:
            continue
        trunk.append(p)
        assigned.add(id(p))
    groups.append(dict(params=trunk, lr=float(cfg.get('lr_backbone', 1e-5)), name='backbone'))

    inject = list(backbone.conv11_head.parameters()) + list(
        backbone.conv22_head.parameters()) + list(backbone.conv33_head.parameters())
    groups.append(dict(
        params=take(inject),
        lr=float(cfg.get('lr_inject', 1e-4)), name='inject'))

    igab = list(backbone.coarse_igab.parameters()) + list(backbone.middle_igab.parameters())
    groups.append(dict(
        params=take(igab),
        lr=float(cfg.get('lr_igab', 2e-5)), name='igab'))

    illum = list(backbone.illumination_estimator.parameters()) + list(
        backbone.merge_tail.parameters())
    groups.append(dict(
        params=take(illum),
        lr=float(cfg.get('lr_illum', 2e-5)), name='illum_tail'))

    # Catch-all: any remaining trainable (should be empty or LTE if unfrozen)
    leftover = take([p for p in model.parameters() if p.requires_grad])
    if leftover:
        groups.append(dict(
            params=leftover,
            lr=float(cfg.get('lr_v3', cfg.get('lr_backbone', 1e-5))),
            name='other'))

    groups = [g for g in groups if len(g['params']) > 0]
    # Sanity: every trainable param exactly once
    n_in_groups = sum(len(g['params']) for g in groups)
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    if n_in_groups != n_train:
        raise RuntimeError(
            'optimizer group coverage mismatch: grouped=%d trainable=%d' % (
                n_in_groups, n_train))
    return torch.optim.Adam(groups, betas=betas, weight_decay=wd)
