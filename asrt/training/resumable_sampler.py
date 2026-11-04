"""Resumable epoch permutation + batch cursor for IRP training.

Trainer owns the *consumed* batch_cursor. Prefetch from num_workers>0 must not
advance this cursor until the batch is actually used in an update.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler[List[int]]):
    """Yields index lists of length batch_size; state is fully serializable."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = True,
        epoch: int = 0,
        batch_cursor: int = 0,
        permutation: Optional[List[int]] = None,
    ):
        if dataset_size < 1:
            raise ValueError('dataset_size must be >= 1')
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = int(epoch)
        self.batch_cursor = int(batch_cursor)
        if permutation is not None:
            self.permutation = [int(i) for i in permutation]
            if len(self.permutation) != self.dataset_size:
                raise ValueError('permutation length mismatch')
        else:
            self.permutation = self._make_permutation(self.epoch)

    def _make_permutation(self, epoch: int) -> List[int]:
        rng = np.random.RandomState(self.seed + int(epoch) * 1000003)
        return rng.permutation(self.dataset_size).tolist()

    def num_batches(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def _batch_starts(self) -> List[int]:
        n = self.dataset_size
        bs = self.batch_size
        if self.drop_last:
            return list(range(0, (n // bs) * bs, bs))
        return list(range(0, n, bs))

    def __len__(self) -> int:
        # Remaining batches in the current epoch from cursor
        return max(0, self.num_batches() - self.batch_cursor)

    def __iter__(self) -> Iterator[List[int]]:
        starts = self._batch_starts()
        # Snapshot cursor for this iterator; advance() is owned by trainer.
        cursor = self.batch_cursor
        for i, start in enumerate(starts):
            if i < cursor:
                continue
            batch = self.permutation[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch

    def peek_batch_indices(self) -> List[int]:
        """Indices of the next batch without advancing."""
        starts = self._batch_starts()
        if self.batch_cursor >= len(starts):
            return []
        start = starts[self.batch_cursor]
        batch = self.permutation[start:start + self.batch_size]
        if self.drop_last and len(batch) < self.batch_size:
            return []
        return batch

    def advance(self, n: int = 1) -> bool:
        """Mark n batches as consumed. Returns True if epoch rolled."""
        rolled = False
        self.batch_cursor += int(n)
        while self.batch_cursor >= self.num_batches():
            self.batch_cursor -= self.num_batches()
            self.epoch += 1
            self.permutation = self._make_permutation(self.epoch)
            rolled = True
        return rolled

    def state_dict(self) -> dict:
        return dict(
            epoch=self.epoch,
            permutation=list(self.permutation),
            batch_cursor=self.batch_cursor,
            seed=self.seed,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            dataset_size=self.dataset_size,
        )

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state['epoch'])
        self.batch_cursor = int(state['batch_cursor'])
        self.seed = int(state['seed'])
        self.batch_size = int(state['batch_size'])
        self.drop_last = bool(state['drop_last'])
        self.dataset_size = int(state['dataset_size'])
        self.permutation = [int(i) for i in state['permutation']]
        if len(self.permutation) != self.dataset_size:
            raise ValueError('loaded permutation size mismatch')
