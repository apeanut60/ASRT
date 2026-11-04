"""IRP window attention unit tests."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from asrt.models.irp_attention import (  # noqa: E402
    IRPTransformerBlock,
    WindowSelfAttention,
    window_partition,
    window_reverse,
)


def test_window_partition_reverse():
    x = torch.randn(2, 64, 32, 40)
    ws = 8
    # pad
    from asrt.models.irp_attention import pad_to_window
    xp, (ph, pw, h, w) = pad_to_window(x, ws)
    wins = window_partition(xp, ws)
    y = window_reverse(wins, ws, xp.shape[-2], xp.shape[-1], 2)
    y = y[:, :, :h, :w]
    assert torch.allclose(x, y, atol=1e-6)


def test_independent_qkvs():
    blk = IRPTransformerBlock(dim=64, heads=4, window_size=8, use_reference=True)
    assert blk.attn_illum is not blk.attn_ref
    assert blk.attn_illum.q is not blk.attn_ref.q
    ids_i = {id(p) for p in blk.attn_illum.parameters()}
    ids_r = {id(p) for p in blk.attn_ref.parameters()}
    assert ids_i.isdisjoint(ids_r)


def test_attention_shape_and_finite():
    attn = WindowSelfAttention(64, 4, 8)
    x = torch.randn(2, 64, 33, 35)  # non-multiple
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_no_ref_branch_when_disabled():
    blk = IRPTransformerBlock(use_reference=False)
    assert blk.attn_ref is None
    f = torch.randn(1, 64, 16, 16)
    p = torch.randn(1, 64, 16, 16)
    y = blk(f, p, None)
    assert y.shape == f.shape


def test_irp_block_dual():
    blk = IRPTransformerBlock(use_reference=True, beta_init=0.1)
    f = torch.randn(1, 64, 24, 24)
    pi = torch.randn(1, 64, 24, 24)
    pr = torch.randn(1, 64, 24, 24)
    y = blk(f, pi, pr)
    assert y.shape == f.shape
    assert torch.isfinite(y).all()


def main():
    test_window_partition_reverse()
    test_independent_qkvs()
    test_attention_shape_and_finite()
    test_no_ref_branch_when_disabled()
    test_irp_block_dual()
    print('ALL test_irp_attention PASS')


if __name__ == '__main__':
    main()
