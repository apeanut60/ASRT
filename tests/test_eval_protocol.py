#!/usr/bin/env python
import unittest

import numpy as np
import torch

from asrt.evaluation.metrics import psnr_rgb, psnr_ssim_rgb, to_uint8_rgb


class TestEvalProtocol(unittest.TestCase):
    def test_identity_psnr(self):
        x = torch.rand(1, 3, 64, 64) * 2 - 1
        m = psnr_ssim_rgb(x, x)
        self.assertGreater(m['psnr_rgb'], 50)
        self.assertGreater(m['ssim_rgb'], 0.99)

    def test_range_uint8(self):
        x = torch.zeros(1, 3, 16, 16)
        u = to_uint8_rgb(x, input_range='minus_one_one')  # 0 → 128
        self.assertEqual(u.dtype, np.uint8)
        self.assertEqual(u.shape[-1], 3)
        self.assertEqual(int(u.mean()), 128)

    def test_explicit_endpoints(self):
        t = torch.stack([
            torch.full((1, 4, 4), -1.0),
            torch.full((1, 4, 4), 0.0),
            torch.full((1, 4, 4), 1.0),
        ], dim=1)
        u = to_uint8_rgb(t, input_range='minus_one_one')
        self.assertEqual(int(u[0, 0, 0]), 0)
        self.assertEqual(int(u[0, 0, 1]), 128)
        self.assertEqual(int(u[0, 0, 2]), 255)

    def test_no_mean_align_needed(self):
        a = np.zeros((32, 32, 3), np.uint8)
        b = np.ones((32, 32, 3), np.uint8) * 10
        p = psnr_rgb(a, b)
        self.assertTrue(np.isfinite(p))


if __name__ == '__main__':
    unittest.main()
