"""Cosine LR schedule."""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, cfg: dict):
    total = int(cfg.get('total_steps', cfg.get('max_steps', 80000)))
    eta_min = float(cfg.get('eta_min', 1e-6))
    base = float(cfg.get('lr', 2e-4))
    # Use ratio relative to each group's base lr
    min_ratio = eta_min / max(base, 1e-12)

    def lr_lambda(step):
        if total <= 1:
            return 1.0
        t = min(max(step, 0), total)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / total))
        return min_ratio + (1.0 - min_ratio) * cos

    # last_epoch=-1: do not advance on construction (avoids step-before-optim warn).
    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
