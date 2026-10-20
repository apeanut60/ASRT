#!/usr/bin/env python
"""Native T_lv* spatial sizes after SearchTransfer."""
import unittest

import torch

from asrt.models.asrt import ASRT


class TestNativeTlv(unittest.TestCase):
    def test_tlv_grids(self):
        m = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True)
        low = torch.randn(1, 3, 64, 64)
        ref = torch.randn(1, 3, 64, 64)
        S, T3, T2, T1, _ = m.match_reference(low, ref)
        # After pad to 64: LTE lv3 is H/4=16, lv2=H/2=32, lv1=H=64
        self.assertEqual(tuple(S.shape[-2:]), (16, 16))
        self.assertEqual(tuple(T3.shape[-2:]), (16, 16))
        self.assertEqual(tuple(T2.shape[-2:]), (32, 32))
        self.assertEqual(tuple(T1.shape[-2:]), (64, 64))
        self.assertEqual(T3.shape[1], 256)
        self.assertEqual(T2.shape[1], 128)
        self.assertEqual(T1.shape[1], 64)


if __name__ == '__main__':
    unittest.main()
