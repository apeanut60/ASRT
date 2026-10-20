#!/usr/bin/env python3
"""Install MainNetCSFIUNet into an existing TTSR-lowlight checkout.

Usage:
  python scripts/install_csfi_unet.py --repo-dir /path/to/TTSR-lowlight

Does not switch the default architecture. Use --mainnet_arch csfi_unet to opt in.
Backs up touched source files to *.before_csfi_unet.bak (once only).
"""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def add_arch_support(repo_dir: Path) -> None:
    target_mod = repo_dir / 'model' / 'MainNetCSFIUNet.py'
    target_mod.parent.mkdir(parents=True, exist_ok=True)
    source_mod = HERE / 'model' / 'MainNetCSFIUNet.py'
    if source_mod.resolve() != target_mod.resolve():
        shutil.copy2(source_mod, target_mod)
    print('module:', target_mod)

    src = repo_dir / 'model' / 'TTSREnhance.py'
    original = src.read_text(encoding='utf-8')
    if 'MainNetCSFIUNet.MainNetCSFIUNet(' not in original:
        needle_import = 'from model import MainNetEnhance, LTE, SearchTransfer'
        if original.count(needle_import) != 1:
            raise RuntimeError('Cannot uniquely locate TTSREnhance model import')
        modified = original.replace(
            needle_import,
            'from model import MainNetEnhance, MainNetCSFIUNet, LTE, SearchTransfer', 1)
        needle_ctor = '''        self.MainNet = MainNetEnhance.MainNetEnhance(
            num_res_blocks=self.num_res_blocks,
            n_feats=args.n_feats,
            res_scale=args.res_scale,
            ref_illum_pool=getattr(args, 'ref_illum_pool', 8))'''
        if modified.count(needle_ctor) != 1:
            raise RuntimeError('Cannot uniquely locate TTSREnhance MainNet ctor')
        new_ctor = '''        mainnet_arch = str(getattr(args, 'mainnet_arch', 'legacy')).lower()
        if mainnet_arch == 'csfi_unet':
            self.MainNet = MainNetCSFIUNet.MainNetCSFIUNet(
                num_res_blocks=self.num_res_blocks,
                n_feats=args.n_feats,
                res_scale=args.res_scale,
                ref_illum_pool=getattr(args, 'ref_illum_pool', 8))
        elif mainnet_arch == 'legacy':
            self.MainNet = MainNetEnhance.MainNetEnhance(
                num_res_blocks=self.num_res_blocks,
                n_feats=args.n_feats,
                res_scale=args.res_scale,
                ref_illum_pool=getattr(args, 'ref_illum_pool', 8))
        else:
            raise ValueError('Unknown --mainnet_arch: %r' % mainnet_arch)'''
        modified = modified.replace(needle_ctor, new_ctor, 1)
        ast.parse(modified, filename=str(src))
        backup = src.with_name(src.name + '.before_csfi_unet.bak')
        if not backup.exists():
            shutil.copy2(src, backup)
        src.write_text(modified, encoding='utf-8')
        print('patched:', src)
    else:
        print('already integrated:', src)

    opt = repo_dir / 'option.py'
    text = opt.read_text(encoding='utf-8')
    if "'--mainnet_arch'" not in text and '"--mainnet_arch"' not in text:
        anchor = "parser.add_argument('--num_res_blocks'"
        if text.count(anchor) != 1:
            raise RuntimeError('Cannot uniquely locate option.py --num_res_blocks')
        addition = ("parser.add_argument('--mainnet_arch', type=str, default='legacy', "
                    "choices=['legacy', 'csfi_unet'],\n"
                    "                    help='Use legacy MainNet or v3 CSFI-U-Net with IGAB')\n")
        updated = text.replace(anchor, addition + anchor, 1)
        ast.parse(updated, filename=str(opt))
        backup = opt.with_name(opt.name + '.before_csfi_unet.bak')
        if not backup.exists():
            shutil.copy2(opt, backup)
        opt.write_text(updated, encoding='utf-8')
        print('patched:', opt)
    else:
        print('already integrated:', opt)
    print('Select new net with --mainnet_arch csfi_unet')
    print('When loading a legacy checkpoint, use --pretrain_require_core False')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo-dir', type=Path, default=Path.cwd())
    args = p.parse_args()
    add_arch_support(args.repo_dir.resolve())
