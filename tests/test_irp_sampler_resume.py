"""ResumableBatchSampler state restore."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.training.resumable_sampler import ResumableBatchSampler  # noqa: E402


def test_resume_continues_permutation():
    s = ResumableBatchSampler(dataset_size=20, batch_size=4, seed=42, drop_last=True)
    assert s.num_batches() == 5
    batches = []
    it = iter(s)
    for _ in range(3):
        b = next(it)
        batches.append(list(b))
        s.advance(1)
    state = s.state_dict()
    assert state['batch_cursor'] == 3

    s2 = ResumableBatchSampler(dataset_size=20, batch_size=4, seed=0)
    s2.load_state_dict(state)
    rest = list(iter(s2))
    # Original remaining from same iterator perspective
    s_full = ResumableBatchSampler(dataset_size=20, batch_size=4, seed=42)
    all_b = [list(b) for b in s_full]
    assert rest == all_b[3:]
    assert batches == all_b[:3]


def test_epoch_roll_changes_perm():
    s = ResumableBatchSampler(dataset_size=8, batch_size=4, seed=1, drop_last=True)
    assert s.num_batches() == 2
    p0 = list(s.permutation)
    s.advance(1)
    s.advance(1)  # roll
    assert s.epoch == 1
    assert s.batch_cursor == 0
    assert list(s.permutation) != p0


def test_state_roundtrip_fields():
    s = ResumableBatchSampler(11, 3, seed=7, drop_last=True)
    s.advance(1)
    st = s.state_dict()
    for k in ('epoch', 'permutation', 'batch_cursor', 'seed', 'batch_size',
              'drop_last', 'dataset_size'):
        assert k in st


def main():
    test_resume_continues_permutation()
    test_epoch_roll_changes_perm()
    test_state_roundtrip_fields()
    print('ALL test_irp_sampler_resume PASS')


if __name__ == '__main__':
    main()
