#!/usr/bin/env python
"""Extra coverage: AMP on CUDA (if available) + metric domain."""
import unittest

import torch

from asrt.evaluation.metrics import psnr_ssim_rgb
from asrt.models.asrt import ASRT


class TestAmpAndMetric(unittest.TestCase):
    def test_metric_finite(self):
        a = torch.rand(1, 3, 32, 32) * 2 - 1
        b = a + 0.1
        m = psnr_ssim_rgb(a, b)
        self.assertTrue(0 < m['psnr_rgb'] < 50)
        self.assertTrue(0 <= m['ssim_rgb'] <= 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_amp_cuda_forward(self):
        m = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True).cuda()
        low = torch.randn(1, 3, 64, 64, device='cuda')
        with torch.cuda.amp.autocast():
            out = m(low, ref=None)
        self.assertTrue(torch.isfinite(out.float()).all())


if __name__ == '__main__':
    unittest.main()
