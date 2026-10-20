"""Minimal fixture validation of the installer and checkpoint migration."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'model'))
from install_csfi_unet import add_arch_support
from warmstart_csfi_unet import build
from MainNetCSFIUNet import MainNetCSFIUNet


class TestInstall(unittest.TestCase):
    def test_installer_patches_known_source_shape_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model').mkdir()
            (root / 'model' / 'TTSREnhance.py').write_text('''from model import MainNetEnhance, LTE, SearchTransfer
class TTSREnhance:
    def __init__(self, args):
        self.num_res_blocks = [0, 1, 1, 1]
        self.MainNet = MainNetEnhance.MainNetEnhance(
            num_res_blocks=self.num_res_blocks,
            n_feats=args.n_feats,
            res_scale=args.res_scale,
            ref_illum_pool=getattr(args, 'ref_illum_pool', 8))
''', encoding='utf-8')
            (root / 'option.py').write_text("parser.add_argument('--num_res_blocks', default='8+8+4+2')\n")
            add_arch_support(root)
            after = (root / 'model' / 'TTSREnhance.py').read_text()
            self.assertIn("mainnet_arch == 'csfi_unet'", after)
            self.assertIn('MainNetCSFIUNet.MainNetCSFIUNet(', after)
            self.assertTrue((root / 'model' / 'MainNetCSFIUNet.py').exists())
            self.assertTrue((root / 'option.py.before_csfi_unet.bak').exists())
            add_arch_support(root)
            self.assertEqual(after, (root / 'model' / 'TTSREnhance.py').read_text())

    def test_warmstart_preserves_old_sfe_and_not_absolute_rgb_head(self):
        torch.manual_seed(42)
        ref = MainNetCSFIUNet(num_res_blocks=[0,1,1,1], n_feats=16)
        old = {'MainNet.'+k:torch.ones_like(v) for k,v in ref.state_dict().items()}
        old['LTE.dummy'] = torch.tensor([17.])
        full,copied=build(old,[0,1,1,1],16)
        self.assertIn('SFE.conv_head.weight',copied)
        self.assertNotIn('merge_tail.conv_tail2.weight',copied)
        self.assertNotIn('conv11_head.weight',copied)
        self.assertTrue(torch.all(full['MainNet.SFE.conv_head.weight'] == 1.))
        self.assertLess(float(full['MainNet.merge_tail.conv_tail2.weight'].abs().max()),1.)
        self.assertTrue(torch.equal(full['LTE.dummy'], torch.tensor([17.])))


if __name__ == '__main__':
    unittest.main()
