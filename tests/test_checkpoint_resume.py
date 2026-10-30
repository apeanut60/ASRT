#!/usr/bin/env python
import os
import tempfile
import unittest

import torch

from asrt.training.trainer import Trainer


class TestCheckpointResume(unittest.TestCase):
    def test_save_load_step(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = dict(
                stage='a', out_dir=td, max_steps=2, total_steps=2,
                batch_size=1, accum=2, lr=1e-4, amp=False, grad_clip=1.0,
                crop=32, log_every=1, eval_every=100, save_every=100,
                workers=0, seed=0, no_ref_prob=1.0,
                model=dict(num_res_blocks='2+2+1+1', n_feats=32, freeze_lte=True),
            )
            # tiny synthetic rows via temp images
            from PIL import Image
            import numpy as np
            from asrt.data.manifest import save_manifest
            imgdir = os.path.join(td, 'img')
            os.makedirs(imgdir)
            rows = []
            for i in range(4):
                lp = os.path.join(imgdir, 'low%04d.png' % i)
                gp = os.path.join(imgdir, 'normal%04d.png' % i)
                Image.fromarray(np.random.randint(0, 40, (48, 48, 3), np.uint8)).save(lp)
                Image.fromarray(np.random.randint(80, 180, (48, 48, 3), np.uint8)).save(gp)
                rows.append(dict(sample_id='low%04d' % i, low_path=lp, gt_path=gp, ref_path=None, domain='real'))
            tr = Trainer(cfg, device='cpu')
            tr.train(train_rows_noref=rows, train_rows_ref=None, eval_rows=None)
            last = os.path.join(td, 'checkpoints', 'last.pt')
            self.assertTrue(os.path.isfile(last))
            step1 = tr.step
            tr2 = Trainer(cfg, device='cpu')
            tr2.load(last, resume_optim=True)
            self.assertEqual(tr2.step, step1)


if __name__ == '__main__':
    unittest.main()
