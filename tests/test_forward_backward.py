#!/usr/bin/env python
import unittest

import torch

from asrt.models.asrt import ASRT


class TestForwardBackward(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.model = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True).to(self.device)

    def test_even_size_grad(self):
        low = torch.randn(1, 3, 64, 64, device=self.device, requires_grad=False)
        ref = torch.randn(1, 3, 64, 64, device=self.device)
        out = self.model(low, ref=ref)
        self.assertEqual(tuple(out.shape), (1, 3, 64, 64))
        loss = out.mean()
        loss.backward()
        # backbone should have grads
        g = next(p.grad for n, p in self.model.backbone.named_parameters() if p.requires_grad and p.grad is not None)
        self.assertTrue(torch.isfinite(g).all())

    def test_odd_size(self):
        low = torch.randn(1, 3, 65, 67)
        out = self.model(low, ref=None)
        self.assertEqual(tuple(out.shape[-2:]), (65, 67))

    def test_amp_cpu_skip(self):
        # AMP primarily for CUDA; ensure forward still works
        out = self.model(torch.randn(1, 3, 48, 48), ref=None)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
