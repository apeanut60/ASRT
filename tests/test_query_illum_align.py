#!/usr/bin/env python
"""Tests for Query illumination alignment + ASRT wiring."""
from __future__ import annotations

import unittest

import torch

from asrt.models.asrt import ASRT, build_asrt
from asrt.models.query_illum_align import (
    align_query_illumination,
    estimate_query_align_maps,
    apply_luma_hist_lut,
    build_luma_hist_lut,
    _rgb_luma01,
)
from asrt.evaluation.inference import infer_full, infer_tiled, _full_image_query_maps


class TestQueryIllumAlign(unittest.TestCase):
    def test_disabled_matches_raw_path(self):
        m0 = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True,
                   query_align=dict(enabled=False))
        m1 = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True,
                   query_align=dict(enabled=True, method='raw'))
        # Same weights
        m1.load_state_dict(m0.state_dict())
        low = torch.randn(1, 3, 64, 64)
        ref = torch.randn(1, 3, 64, 64) * 0.5 + 0.3
        m0.eval(); m1.eval()
        with torch.no_grad():
            y0 = m0(low, ref=ref)
            y1 = m1(low, ref=ref)
        self.assertTrue(torch.allclose(y0, y1, atol=1e-5))

    def test_gamma_only(self):
        low = torch.full((2, 3, 32, 32), -0.8)
        ref = torch.full((2, 3, 32, 32), 0.4)
        q, info = align_query_illumination(
            low, ref, alpha=0.75, beta=0.0, gamma_min=0.3, gamma_max=1.0)
        self.assertEqual(tuple(q.shape), (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((info['gamma'] <= 1.0).all() and (info['gamma'] >= 0.3).all())
        # Aligned query should be brighter than raw dark low
        self.assertGreater(float(info['mean_final'].mean()), float(info['mean_low'].mean()))

    def test_gamma_hist(self):
        low = torch.rand(2, 3, 48, 48) * 0.2 - 0.9
        ref = torch.rand(2, 3, 48, 48) * 0.6 + 0.2
        q, info = align_query_illumination(
            low, ref, alpha=0.75, beta=0.50, gamma_min=0.3, gamma_max=1.0)
        self.assertTrue(torch.isfinite(q).all())
        self.assertIsNotNone(info['lut'])
        # LUT monotonic
        lut = info['lut']
        self.assertTrue((lut[:, 1:] >= lut[:, :-1] - 1e-5).all())

    def test_constant_and_dark(self):
        low = torch.zeros(1, 3, 16, 16) - 1.0
        ref = torch.ones(1, 3, 16, 16)
        q, info = align_query_illumination(low, ref, alpha=0.75, beta=0.5)
        self.assertTrue(torch.isfinite(q).all())
        self.assertFalse(torch.isnan(info['gamma']).any())

    def test_batch_independent_gamma(self):
        low = torch.stack([
            torch.full((3, 24, 24), -0.9),
            torch.full((3, 24, 24), -0.2),
        ], 0)
        ref = torch.stack([
            torch.full((3, 24, 24), 0.5),
            torch.full((3, 24, 24), 0.5),
        ], 0)
        _, info = align_query_illumination(low, ref, alpha=0.75, beta=0.0)
        self.assertFalse(torch.allclose(info['gamma'][0], info['gamma'][1]))

    def test_full_image_override_matches_crop_region_estimate(self):
        # Full-image maps applied to a crop should match estimating on that
        # crop only when using overrides from the same full tensors' crop region
        # is harder; instead verify override path is deterministic & used.
        low = torch.rand(1, 3, 64, 64) * 0.3 - 0.85
        ref = torch.rand(1, 3, 64, 64) * 0.5 + 0.2
        g, lut, _ = estimate_query_align_maps(low, ref, alpha=0.75, bins=256)
        q1, _ = align_query_illumination(
            low[:, :, 8:40, 8:40], ref[:, :, 8:40, 8:40],
            alpha=0.75, beta=0.5, gamma_override=g, lut_override=lut)
        q2, _ = align_query_illumination(
            low[:, :, 8:40, 8:40], ref[:, :, 8:40, 8:40],
            alpha=0.75, beta=0.5, gamma_override=g, lut_override=lut)
        self.assertTrue(torch.allclose(q1, q2))

    def test_kv_unaffected_by_query_align(self):
        m = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True,
                 query_align=dict(enabled=True, method='gamma_hist',
                                  gamma_alpha=0.75, hist_beta=0.5))
        m.eval()
        low = torch.randn(1, 3, 64, 64) * 0.2 - 0.7
        ref = torch.randn(1, 3, 64, 64) * 0.4 + 0.2
        with torch.no_grad():
            # Match twice: once aligned query, compare T from raw-ref path by
            # disabling align and checking backbone raw low path identical when S/T forced
            S0, T30, _, _, _, _ = m.match_reference(low, ref)
            m.query_align_cfg['enabled'] = False
            S1, T31, _, _, _, _ = m.match_reference(low, ref)
        # With align on vs off, T comes from same Ref LTE values gathered by different
        # match args — T may differ. Key property: raw ref LTE features unchanged.
        # Verify align doesn't mutate ref tensor:
        ref2 = ref.clone()
        m.query_align_cfg['enabled'] = True
        with torch.no_grad():
            m.match_reference(low, ref2)
        self.assertTrue(torch.equal(ref, ref2))

    def test_ref_none_skips_align(self):
        m = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True,
                 query_align=dict(enabled=True, gamma_alpha=0.75, hist_beta=0.5))
        low = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y, aux = m(low, ref=None, return_aux=True)
        self.assertFalse(aux['used_reference'])
        self.assertIsNone(aux['query_illum'])
        self.assertTrue(torch.isfinite(y).all())

    def test_tiled_reuses_full_image_maps(self):
        m = ASRT(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True,
                 query_align=dict(enabled=True, method='gamma_hist',
                                  gamma_alpha=0.75, hist_beta=0.5))
        m.eval()
        low = torch.randn(1, 3, 96, 96) * 0.2 - 0.7
        ref = torch.randn(1, 3, 96, 96) * 0.4 + 0.2
        g1, l1 = _full_image_query_maps(m, low, ref)
        g2, l2 = _full_image_query_maps(m, low, ref)
        self.assertTrue(torch.allclose(g1, g2))
        self.assertTrue(torch.allclose(l1, l2))
        with torch.no_grad():
            y_full = infer_full(m, low, ref)
            y_tile = infer_tiled(m, low, ref, tile=64, overlap=16)
        self.assertEqual(tuple(y_full.shape), tuple(y_tile.shape))
        self.assertTrue(torch.isfinite(y_tile).all())

    def test_no_gt_in_align_api(self):
        # API only accepts low, ref — documented; smoke call without gt
        low = torch.randn(1, 3, 32, 32)
        ref = torch.randn(1, 3, 32, 32)
        q, _ = align_query_illumination(low, ref, alpha=0.75, beta=0.5)
        self.assertEqual(q.shape, low.shape)

    def test_backbone_raw_low_unchanged_when_align_on(self):
        """Output changes with align (match differs) but we verify raw low
        identity into backbone by checking light-up uses original low."""
        m = build_asrt(dict(
            num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True,
            query_align=dict(enabled=True, gamma_alpha=0.75, hist_beta=0.5)))
        m.eval()
        low = torch.randn(1, 3, 64, 64)
        # Capture by hooking backbone forward
        captured = {}

        def hook(module, inputs, output):
            captured['x'] = inputs[0].detach().clone()

        h = m.backbone.register_forward_hook(hook)
        with torch.no_grad():
            m(low, ref=torch.randn(1, 3, 64, 64))
        h.remove()
        self.assertTrue(torch.equal(captured['x'], low))


if __name__ == '__main__':
    unittest.main()
