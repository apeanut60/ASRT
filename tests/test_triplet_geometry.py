#!/usr/bin/env python
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from asrt.data.paired import sync_geometry
from asrt.data.triplet import TripletRefDataset
import torch


class TestTripletGeometry(unittest.TestCase):
    def test_sync_same_crop(self):
        rng = np.random.RandomState(0)
        a = torch.arange(3 * 40 * 40, dtype=torch.float32).view(3, 40, 40)
        b = a.clone()
        c = a.clone()
        aa, bb, cc = sync_geometry([a, b, c], crop=16, rng=rng)
        self.assertEqual(aa.shape[-2:], (16, 16))
        self.assertTrue(torch.equal(aa, bb))
        self.assertTrue(torch.equal(bb, cc))

    def test_dataset_requires_ref(self):
        with tempfile.TemporaryDirectory() as td:
            low = os.path.join(td, 'low.png')
            gt = os.path.join(td, 'normal.png')
            Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(low)
            Image.fromarray(np.ones((32, 32, 3), np.uint8) * 10).save(gt)
            rows = [dict(sample_id='x', low_path=low, gt_path=gt, ref_path=None)]
            with self.assertRaises(SystemExit):
                TripletRefDataset(rows, crop=16, allow_missing_ref=False)


if __name__ == '__main__':
    unittest.main()
