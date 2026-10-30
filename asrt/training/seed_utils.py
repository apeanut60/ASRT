"""Deterministic seeding helpers for ASRT training."""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive per-worker RNG from torch initial seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def capture_rng_state() -> Dict[str, Any]:
    state = dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    if state.get('python') is not None:
        random.setstate(state['python'])
    if state.get('numpy') is not None:
        np.random.set_state(state['numpy'])
    if state.get('torch') is not None:
        torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
