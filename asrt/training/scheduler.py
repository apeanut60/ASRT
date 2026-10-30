"""Cosine LR schedule with absolute eta_min per param group.

Semantics (post-trainfix):
  Each param group with base LR `lr_g` decays as
      lr(t) = eta_min + (lr_g - eta_min) * 0.5 * (1 + cos(pi * t / T))
  so every group approaches the same absolute floor `eta_min`.

  Previously a single min_ratio = eta_min / cfg['lr'] was applied to all groups,
  so a group with lr_v4=1e-4 and eta_min=1e-6 (with cfg lr=1e-5) would floor at
  1e-5 (= 0.1 * 1e-4), not 1e-6. That relative behavior is retired.
"""
from __future__ import annotations

import math
from typing import List

from torch.optim.lr_scheduler import LambdaLR


class AbsoluteCosineScheduler:
    """Thin wrapper: expose .step/.state_dict like LambdaLR, absolute eta_min."""

    def __init__(self, optimizer, total_steps: int, eta_min: float):
        self.optimizer = optimizer
        self.total = max(int(total_steps), 1)
        self.eta_min = float(eta_min)
        self.last_epoch = -1
        # Snapshot base LRs at construction (post-optimizer build).
        # Do not advance on construction — first step() applies t=0 cosine factor.
        self.base_lrs: List[float] = [float(g['lr']) for g in optimizer.param_groups]

    def _factor(self, step: int) -> float:
        t = min(max(int(step), 0), self.total)
        if self.total <= 1:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * t / self.total))

    def _apply(self, step: int) -> None:
        cos = self._factor(step)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            # If base already <= eta_min (e.g. zeroed during stable), keep as-is floor
            if base <= self.eta_min:
                g['lr'] = base * cos  # usually 0
            else:
                g['lr'] = self.eta_min + (base - self.eta_min) * cos

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)
        self._apply(self.last_epoch)

    def state_dict(self):
        return dict(
            last_epoch=self.last_epoch,
            total=self.total,
            eta_min=self.eta_min,
            base_lrs=list(self.base_lrs),
        )

    def load_state_dict(self, state):
        self.last_epoch = int(state.get('last_epoch', -1))
        self.total = int(state.get('total', self.total))
        self.eta_min = float(state.get('eta_min', self.eta_min))
        if state.get('base_lrs') is not None:
            self.base_lrs = [float(x) for x in state['base_lrs']]
        self._apply(max(self.last_epoch, 0) if self.last_epoch >= 0 else 0)

    def get_last_lr(self):
        return [float(g['lr']) for g in self.optimizer.param_groups]


def build_scheduler(optimizer, cfg: dict):
    total = int(cfg.get('total_steps', cfg.get('max_steps', 80000)))
    eta_min = float(cfg.get('eta_min', 1e-6))
    return AbsoluteCosineScheduler(optimizer, total_steps=total, eta_min=eta_min)


def preview_lrs(optimizer, cfg: dict, fractions=(0.0, 0.5, 1.0)) -> dict:
    """Report each group's LR at given fractions of total_steps (diagnostic)."""
    total = int(cfg.get('total_steps', cfg.get('max_steps', 80000)))
    eta_min = float(cfg.get('eta_min', 1e-6))
    bases = [float(g['lr']) for g in optimizer.param_groups]
    names = [g.get('name', str(i)) for i, g in enumerate(optimizer.param_groups)]
    out = {}
    for frac in fractions:
        step = int(round(frac * total))
        if total <= 1:
            cos = 1.0
        else:
            t = min(max(step, 0), total)
            cos = 0.5 * (1.0 + math.cos(math.pi * t / total))
        row = {}
        for name, base in zip(names, bases):
            if base <= eta_min:
                row[name] = base * cos
            else:
                row[name] = eta_min + (base - eta_min) * cos
        out['frac_%.2f_step_%d' % (frac, step)] = row
    return out
