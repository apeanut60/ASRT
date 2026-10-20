#!/usr/bin/env python
import unittest

import torch

from asrt.models.asrt import ASRT


class TestReferenceBypass(unittest.TestCase):
    def test_noref_skips_match(self):
        m = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True)
        low = torch.randn(1, 3, 64, 64)
        out, aux = m(low, ref=None, return_aux=True)
        self.assertFalse(aux['used_reference'])
        self.assertIsNone(aux['S'])
        self.assertIsNone(aux['gate'])
        self.assertEqual(tuple(out.shape), (1, 3, 64, 64))

    def test_ref_changes_output(self):
        m = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True)
        # Soft inject init ≈0; amplify so different refs diverge.
        with torch.no_grad():
            for head in (m.backbone.conv11_head, m.backbone.conv22_head,
                         m.backbone.conv33_head):
                head.weight.mul_(100.0)
        m.eval()
        low = torch.randn(1, 3, 64, 64)
        ref_a = torch.randn(1, 3, 64, 64)
        ref_b = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            ya, aa = m(low, ref=ref_a, return_aux=True)
            yb, ab = m(low, ref=ref_b, return_aux=True)
            y0 = m(low, ref=None)
        self.assertTrue(aa['used_reference'] and ab['used_reference'])
        self.assertFalse(torch.allclose(ya, yb, atol=1e-4))
        self.assertTrue(torch.isfinite(y0).all())

    def test_s_none_safe_on_backbone(self):
        from asrt.models.csfi_unet import CSFIUNet
        b = CSFIUNet(num_res_blocks='2+2+1+1', n_feats=32)
        x = torch.randn(1, 3, 64, 64)
        y = b(x, S=None, T_lv3=None, use_reference=False)
        self.assertEqual(tuple(y.shape), (1, 3, 64, 64))


if __name__ == '__main__':
    unittest.main()
