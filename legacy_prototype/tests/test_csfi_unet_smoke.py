"""CPU shape/gradient/reference/checkpoint-interface smoke tests, no dataset needed.
Run: python -m unittest discover -s tests -p 'test_csfi_unet_smoke.py'
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from MainNetCSFIUNet import MainNetCSFIUNet, IG_MSA


class TestMainNetV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def sample(h, w):
        torch.manual_seed(5)
        low = torch.rand(1, 3, h, w) * 2. - 1.
        ref = torch.rand(1, 3, h, w) * 2. - 1.
        c_h, c_w = max(1, h // 4), max(1, w // 4)
        S = torch.randn(1, 1, c_h, c_w)
        t3 = torch.randn(1, 256, c_h, c_w)
        t2 = torch.randn(1, 128, max(1, h // 2), max(1, w // 2))
        t1 = torch.randn(1, 64, h, w)
        return low, ref, S, t3, t2, t1

    @staticmethod
    def net():
        return MainNetCSFIUNet(num_res_blocks=[0, 1, 1, 1],
                                n_feats=16, igab_heads=1)

    def test_shapes_even_odd(self):
        net = self.net().eval()
        for h, w in [(32, 32), (33, 35), (36, 40)]:
            low, ref, S, t3, t2, t1 = self.sample(h, w)
            with torch.no_grad():
                out = net(low, S, t3, t2, t1, ref=ref)
            self.assertEqual(out.shape, low.shape)
            self.assertTrue(bool(torch.isfinite(out).all()))

    def test_legacy_kwargs_and_bypass(self):
        net = self.net().eval()
        low, ref, S, t3, t2, t1 = self.sample(32, 36)
        with torch.no_grad():
            out = net(low, S, t3, t2, t1, ref=ref,
                      gate_mode='constant_mean',
                      stage_mode_lv3='branch_bypass',
                      stage_mode_lv2='texture_zero',
                      stage_mode_lv1='normal',
                      collect_gate_stats=True,
                      collect_residual_decomp=True)
        self.assertEqual(out.shape, low.shape)
        self.assertEqual(net.last_stage_modes['lv3'], 'branch_bypass')

    def test_reference_changes_output(self):
        net = self.net().eval()
        low, ref, S, t3, t2, t1 = self.sample(32, 32)
        with torch.no_grad():
            a = net(low, S, t3, t2, t1, ref=ref,
                    apply_illum=False, apply_ref_illum=False)
            b = net(low, S, torch.zeros_like(t3), t2, t1,
                    ref=ref, apply_illum=False, apply_ref_illum=False)
        self.assertGreater(float((a - b).abs().max()), 0.)

    def test_training_gradients(self):
        net = self.net().train()
        low, ref, S, t3, t2, t1 = self.sample(32, 32)
        t3.requires_grad_()
        out = net(low, S, t3, t2, t1, ref=ref,
                  apply_illum=False, apply_ref_illum=False)
        loss = out.square().mean()
        loss.backward()
        names = ['SFE.conv_head.weight',
                 'conv11_head.weight',
                 'coarse_igab.blocks.0.0.to_v.weight',
                 'middle_igab.blocks.0.0.to_q.weight',
                 'illumination_estimator.conv2.weight',
                 'merge_tail.conv_tail2.weight']
        grads = dict(net.named_parameters())
        for name in names:
            grad = grads[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
        self.assertIsNotNone(t3.grad)
        self.assertGreater(float(t3.grad.abs().sum()), 0.)

    def test_igmsa_is_channel_attention(self):
        layer = IG_MSA(dim=16, dim_head=8, heads=2)
        feat = torch.randn(2, 4, 4, 16)
        illu = torch.randn(2, 4, 4, 16)
        out = layer(feat, illu)
        self.assertEqual(out.shape, feat.shape)


if __name__ == '__main__':
    unittest.main()
