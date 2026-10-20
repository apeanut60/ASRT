"""ASRT training loop with AMP, clip, checkpoint resume."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from asrt.data.paired import PairedRestoreDataset
from asrt.data.triplet import TripletRefDataset, collate_skip_none_ref
from asrt.evaluation.metrics import psnr_ssim_rgb
from asrt.models.asrt import ASRT, build_asrt
from asrt.training.optimizer import build_optimizer
from asrt.training.scheduler import build_scheduler


def l1_loss(pred, gt):
    return F.l1_loss(pred, gt)


class Trainer:
    def __init__(self, cfg: dict, device: Optional[str] = None):
        self.cfg = dict(cfg)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_asrt(cfg.get('model', {})).to(self.device)
        self.opt = build_optimizer(self.model, cfg)
        self.sched = build_scheduler(self.opt, cfg)
        self.scaler = GradScaler(enabled=bool(cfg.get('amp', True)) and self.device == 'cuda')
        self.clip = float(cfg.get('grad_clip', 1.0))
        self.step = 0
        self.out_dir = cfg.get('out_dir', 'outputs/run')
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)
        self.log_f = None
        self.best = None
        self.no_ref_prob = float(cfg.get('no_ref_prob', 0.0))

    def _loader(self, rows, crop, with_ref: bool, shuffle=True):
        bs = int(self.cfg.get('batch_size', 1))
        workers = int(self.cfg.get('workers', 2))
        if with_ref:
            ds = TripletRefDataset(
                rows, crop=crop, seed=int(self.cfg.get('seed', 42)),
                augment=True, allow_missing_ref=False)
        else:
            ds = PairedRestoreDataset(
                rows, crop=crop, seed=int(self.cfg.get('seed', 42)), augment=True)
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
            drop_last=True, collate_fn=collate_skip_none_ref)

    def save(self, tag: str, extra: Optional[dict] = None):
        path = os.path.join(self.out_dir, 'checkpoints', '%s.pt' % tag)
        payload = dict(
            model=self.model.state_dict(),
            optimizer=self.opt.state_dict(),
            scheduler=self.sched.state_dict(),
            scaler=self.scaler.state_dict(),
            step=self.step,
            cfg=self.cfg,
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            best=self.best,
            extra=extra or {},
        )
        torch.save(payload, path)
        return path

    def load(self, path: str, resume_optim: bool = True):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck['model'])
        if resume_optim:
            self.step = int(ck.get('step', 0))
            self.best = ck.get('best')
            if 'optimizer' in ck:
                self.opt.load_state_dict(ck['optimizer'])
            if 'scheduler' in ck:
                self.sched.load_state_dict(ck['scheduler'])
            if 'scaler' in ck:
                self.scaler.load_state_dict(ck['scaler'])
        else:
            # Warm-start weights only; start a fresh schedule.
            self.step = 0
            self.best = None
        print('loaded %s @ step %d (resume_optim=%s)' % (
            path, self.step, resume_optim), flush=True)

    @torch.no_grad()
    def evaluate(self, rows, with_ref: bool, limit: int = 0):
        from asrt.data.paired import read_rgb_m11
        from asrt.evaluation.inference import infer_full, infer_tiled

        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scores = []
        n = len(rows) if not limit else min(limit, len(rows))
        tile = int(self.cfg.get('eval_tile', 256))
        max_full = int(self.cfg.get('eval_full_max_side', 512))
        for i in range(n):
            row = rows[i]
            low = read_rgb_m11(row['low_path'])[None].to(self.device)
            gt = read_rgb_m11(row['gt_path'])[None].to(self.device)
            if tuple(gt.shape[-2:]) != tuple(low.shape[-2:]):
                gt = F.interpolate(gt, size=low.shape[-2:], mode='bilinear', align_corners=False)
            ref = None
            if with_ref and row.get('ref_path'):
                ref = read_rgb_m11(row['ref_path'])[None].to(self.device)
                if tuple(ref.shape[-2:]) != tuple(low.shape[-2:]):
                    ref = F.interpolate(ref, size=low.shape[-2:], mode='bilinear', align_corners=False)
            use_tile = max(low.shape[-2:]) > max_full
            if use_tile:
                pred = infer_tiled(self.model, low, ref, tile=tile, overlap=32)
            else:
                pred = infer_full(self.model, low, ref)
            scores.append(psnr_ssim_rgb(pred, gt))
            del low, gt, ref, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not scores:
            return dict(psnr_rgb=0.0, ssim_rgb=0.0, n=0)
        return dict(
            psnr_rgb=float(np.mean([s['psnr_rgb'] for s in scores])),
            ssim_rgb=float(np.mean([s['ssim_rgb'] for s in scores])),
            n=len(scores),
        )

    def train(self, train_rows_ref=None, train_rows_noref=None, eval_rows=None):
        max_steps = int(self.cfg.get('max_steps', self.cfg.get('total_steps', 1000)))
        log_every = int(self.cfg.get('log_every', 20))
        eval_every = int(self.cfg.get('eval_every', 200))
        save_every = int(self.cfg.get('save_every', eval_every))
        crop_ref = int(self.cfg.get('crop', 256))
        crop_noref = int(self.cfg.get('crop_noref', crop_ref))
        stage = str(self.cfg.get('stage', 'a')).lower()
        use_ref_default = stage not in ('a', 'stage_a')

        # Build infinite iterators. Stage A may split Real/Syn crops.
        crop_syn = int(self.cfg.get('crop_syn', crop_noref))
        loaders = {}
        if train_rows_noref:
            real_rows = [r for r in train_rows_noref
                         if str(r.get('domain', '')).lower() == 'real'
                         or '/lol-v2-real/' in str(r.get('low_path', ''))]
            syn_rows = [r for r in train_rows_noref
                        if str(r.get('domain', '')).lower() == 'syn'
                        or 'synthetic' in str(r.get('low_path', '')).lower()]
            if real_rows and syn_rows and crop_syn != crop_noref:
                loaders['noref_real'] = self._loader(real_rows, crop_noref, with_ref=False)
                loaders['noref_syn'] = self._loader(syn_rows, crop_syn, with_ref=False)
            else:
                loaders['noref'] = self._loader(train_rows_noref, crop_noref, with_ref=False)
        if train_rows_ref and use_ref_default:
            loaders['ref'] = self._loader(train_rows_ref, crop_ref, with_ref=True)
        if not loaders:
            raise SystemExit('no training rows')

        iters = {k: iter(v) for k, v in loaders.items()}
        epoch = {k: 0 for k in loaders}

        def next_batch(kind):
            try:
                return next(iters[kind])
            except StopIteration:
                epoch[kind] += 1
                if hasattr(loaders[kind].dataset, 'set_epoch'):
                    loaders[kind].dataset.set_epoch(epoch[kind])
                iters[kind] = iter(loaders[kind])
                return next(iters[kind])

        noref_keys = [k for k in loaders if k.startswith('noref')]

        t0 = time.time()
        peak_mem = 0.0
        clipped = 0
        self.log_f = open(os.path.join(self.out_dir, 'train.jsonl'), 'a')
        self.model.train()
        while self.step < max_steps:
            # Choose ref / no-ref
            if 'ref' in loaders and noref_keys:
                use_ref = (np.random.rand() >= self.no_ref_prob)
            elif 'ref' in loaders:
                use_ref = True
            else:
                use_ref = False
            if use_ref:
                batch = next_batch('ref')
            else:
                kind = noref_keys[int(np.random.randint(0, len(noref_keys)))]
                batch = next_batch(kind)
            low = batch['low'].to(self.device)
            gt = batch['gt'].to(self.device)
            ref = None if (not use_ref or batch.get('ref') is None) else batch['ref'].to(self.device)

            self.opt.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                pred = self.model(low, ref=ref)
                loss = l1_loss(pred, gt)
            if not torch.isfinite(loss):
                raise SystemExit('non-finite loss at step %d' % self.step)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            if self.clip > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.clip)
                if float(gnorm) > self.clip:
                    clipped += 1
            else:
                gnorm = 0.0
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.step += 1

            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))

            if self.step % log_every == 0 or self.step == 1:
                lr = self.opt.param_groups[0]['lr']
                row = dict(
                    step=self.step, loss=float(loss.detach()), lr=lr,
                    gnorm=float(gnorm), used_ref=bool(use_ref),
                    peak_mem_mb=peak_mem, clip_events=clipped,
                )
                self.log_f.write(json.dumps(row) + '\n')
                self.log_f.flush()
                print('step %d/%d loss=%.4f lr=%.2e ref=%s mem=%.0fMB' % (
                    self.step, max_steps, row['loss'], lr, use_ref, peak_mem), flush=True)

            if eval_rows and (self.step % eval_every == 0 or self.step == max_steps):
                metrics = self.evaluate(
                    eval_rows, with_ref=use_ref_default and any(r.get('ref_path') for r in eval_rows),
                    limit=int(self.cfg.get('eval_limit', 0)))
                print('  eval PSNR=%.4f SSIM=%.4f n=%d' % (
                    metrics['psnr_rgb'], metrics['ssim_rgb'], metrics['n']), flush=True)
                if self.best is None or metrics['psnr_rgb'] > self.best['psnr_rgb']:
                    self.best = dict(step=self.step, **metrics)
                    self.save('best', extra=metrics)
                self.model.train()

            if self.step % save_every == 0:
                self.save('step_%06d' % self.step)

        self.save('last')
        summary = dict(
            steps=self.step,
            seconds=time.time() - t0,
            peak_mem_mb=peak_mem,
            clip_events=clipped,
            best=self.best,
            cfg=self.cfg,
        )
        with open(os.path.join(self.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        self.log_f.close()
        print('=== done in %.1f min ===' % (summary['seconds'] / 60), flush=True)
        return summary
