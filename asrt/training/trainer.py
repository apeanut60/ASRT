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
from asrt.models.asrt import ASRT, build_asrt, load_pretrained_into_v3
from asrt.models.asrt_v4 import ASRTv4, build_asrt_v4, load_stage_a_into_v4
from asrt.training.optimizer import build_optimizer
from asrt.training.scheduler import build_scheduler


def l1_loss(pred, gt):
    return F.l1_loss(pred, gt)


def _is_v4_cfg(cfg: dict, model_cfg: dict) -> bool:
    if bool(model_cfg.get('enable_v4', False)):
        return True
    ver = str(model_cfg.get('version', '')).lower()
    if ver in ('v4', 'asrt_v4', 'asrt-v4'):
        return True
    stage = str(cfg.get('stage', '')).lower()
    return stage in ('b_v4', 'stage_b_v4', 'v4')


class Trainer:
    def __init__(self, cfg: dict, device: Optional[str] = None):
        self.cfg = dict(cfg)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_cfg = dict(cfg.get('model', {}) or {})
        # Top-level query_alignment merges into model.query_align
        if cfg.get('query_alignment') is not None:
            model_cfg['query_align'] = dict(cfg['query_alignment'])
        self.is_v4 = _is_v4_cfg(cfg, model_cfg)
        if self.is_v4:
            self.model = build_asrt_v4(model_cfg).to(self.device)
        else:
            self.model = build_asrt(model_cfg).to(self.device)
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
        self.lambda_no_ref = float(cfg.get('lambda_no_ref', 0.0))
        self.dual_forward = bool(cfg.get('dual_forward', self.is_v4))
        self.teacher_w = float(cfg.get('teacher_consistency_weight', 0.0))
        self.stable_steps = int(cfg.get('stable_steps', 0))
        self._base_lrs = [float(g['lr']) for g in self.opt.param_groups]
        self.teacher = None
        t_ckpt = cfg.get('teacher_ckpt') or ''
        if (not self.is_v4) and self.teacher_w > 0 and t_ckpt and os.path.isfile(t_ckpt):
            # Frozen Stage-A teacher (no v3); eval mode forever — not used for v4
            t_cfg = dict(model_cfg)
            t_cfg['enable_v3'] = False
            self.teacher = build_asrt(t_cfg).to(self.device)
            ck = torch.load(t_ckpt, map_location=self.device)
            state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
            self.teacher.load_state_dict(state, strict=True)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            print('teacher loaded from %s (w=%.3f)' % (t_ckpt, self.teacher_w), flush=True)

    def _apply_stable_lrs(self):
        """During stable_steps: keep v3_new/v4_new LR, zero other group LRs."""
        if self.stable_steps <= 0:
            return
        in_stable = self.step < self.stable_steps
        if not in_stable:
            return
        keep = {'v3_new', 'v4_new'}
        for g, base in zip(self.opt.param_groups, self._base_lrs):
            if g.get('name') in keep:
                g['lr'] = base
            else:
                g['lr'] = 0.0

    def _loader(self, rows, crop, with_ref: bool, shuffle=True):
        bs = int(self.cfg.get('batch_size', 1))
        workers = int(self.cfg.get('workers', 2))
        qa = self.cfg.get('query_alignment') or self.cfg.get('model', {}).get('query_align')
        if with_ref:
            ds = TripletRefDataset(
                rows, crop=crop, seed=int(self.cfg.get('seed', 42)),
                augment=True, allow_missing_ref=False, query_align=qa)
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
        state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        if (self.is_v4 and isinstance(self.model, ASRTv4) and not resume_optim
                and bool(self.cfg.get('pretrained_into_v4', True))):
            info = load_stage_a_into_v4(self.model, state)
            print(
                'v4 warm-start from Stage A: loaded=%d new=%d unused_ckpt=%d'
                % (info['loaded'], len(info['expected_new']), len(info['unused_ckpt'])),
                flush=True)
            report_path = os.path.join(self.out_dir, 'stage_a_load_report.json')
            with open(report_path, 'w') as f:
                json.dump(info, f, indent=2)
            print('load report → %s' % report_path, flush=True)
        elif (getattr(self.model, 'enable_v3', False)
                and not resume_optim
                and bool(self.cfg.get('pretrained_into_v3',
                                      self.cfg.get('pretrained_is_b_v2', True)))):
            info = load_pretrained_into_v3(self.model, state)
            print('v3 warm-start from pretrained: loaded=%d missing_new=%d' % (
                info['loaded'], len(info['missing'])), flush=True)
        else:
            self.model.load_state_dict(state, strict=True)
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
                pred = infer_full(self.model, low, ref)  # full-image align inside model
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

        accum = max(1, int(self.cfg.get('accum', 1)))
        micro_bs = int(self.cfg.get('batch_size', 1))
        eff_batch = micro_bs * accum
        print('train accum=%d batch=%d effective_batch=%d' % (
            accum, micro_bs, eff_batch), flush=True)

        t0 = time.time()
        peak_mem = 0.0
        clipped = 0
        sample_stats = dict(
            ref_batches=0, pure_noref_batches=0,
            dual_noref_loss_sum=0.0, pure_noref_loss_sum=0.0,
            dual_noref_n=0, pure_noref_n=0,
            domain_real=0, domain_syn=0, domain_other=0,
        )
        print(
            'sampling: no_ref_prob=%.3f dual_forward=%s lambda_no_ref=%.3f'
            % (self.no_ref_prob, self.dual_forward, self.lambda_no_ref),
            flush=True)
        self.log_f = open(os.path.join(self.out_dir, 'train.jsonl'), 'a')
        self.model.train()
        while self.step < max_steps:
            self.opt.zero_grad(set_to_none=True)
            loss_avg = 0.0
            use_ref = False
            for _micro in range(accum):
                # Choose ref / no-ref per micro-batch
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
            qg = batch.get('query_gamma')
            ql = batch.get('query_lut')
            if qg is not None:
                qg = qg.to(self.device)
            if ql is not None:
                ql = ql.to(self.device)

            # Domain accounting (batch may carry per-sample domain via dataset rows)
            domains = batch.get('domain')
            if domains is None:
                sample_stats['domain_other'] += 1
            else:
                if torch.is_tensor(domains):
                    domains = [str(d) for d in domains]
                elif isinstance(domains, str):
                    domains = [domains]
                for d in domains:
                    dl = str(d).lower()
                    if dl == 'real':
                        sample_stats['domain_real'] += 1
                    elif dl in ('syn', 'synthetic'):
                        sample_stats['domain_syn'] += 1
                    else:
                        sample_stats['domain_other'] += 1

            # Stable period: only train new v3 modules initially
            self._apply_stable_lrs()

            teacher_l = 0.0
            loss_ref_v = 0.0
            loss_noref_v = 0.0
            loss_pure_noref_v = 0.0
            batch_kind = 'ref_dual' if (use_ref and ref is not None) else 'pure_noref'
            with autocast(enabled=self.scaler.is_enabled()):
                if self.dual_forward and use_ref and ref is not None:
                    # ASRT-v4: same Low/GT with and without reference
                    pred_ref = self.model(low, ref=ref, query_gamma=qg, query_lut=ql)
                    aux_ref = getattr(self.model, 'last_aux', None)
                    pred_noref = self.model(low, ref=None)
                    # Keep ref-path diagnostics for logging (no-ref overwrites last_aux)
                    if aux_ref is not None:
                        self.model.last_aux = aux_ref
                    loss_ref = l1_loss(pred_ref, gt)
                    loss_noref = l1_loss(pred_noref, gt)
                    loss = loss_ref + self.lambda_no_ref * loss_noref
                    loss_ref_v = float(loss_ref.detach())
                    loss_noref_v = float(loss_noref.detach())
                    sample_stats['ref_batches'] += 1
                    sample_stats['dual_noref_loss_sum'] += loss_noref_v
                    sample_stats['dual_noref_n'] += 1
                    pred = pred_ref
                else:
                    pred = self.model(low, ref=ref, query_gamma=qg, query_lut=ql)
                    loss = l1_loss(pred, gt)
                    loss_pure_noref_v = float(loss.detach())
                    sample_stats['pure_noref_batches'] += 1
                    sample_stats['pure_noref_loss_sum'] += loss_pure_noref_v
                    sample_stats['pure_noref_n'] += 1
                    batch_kind = 'pure_noref'
                    if (self.teacher is not None and self.teacher_w > 0
                            and (not use_ref or ref is None)):
                        with torch.no_grad():
                            t_pred = self.teacher(low, ref=None)
                        t_loss = l1_loss(pred, t_pred)
                        loss = loss + self.teacher_w * t_loss
                        teacher_l = float(t_loss.detach())
                loss = loss / float(accum)
                if not torch.isfinite(loss):
                    raise SystemExit('non-finite loss at step %d' % self.step)
                self.scaler.scale(loss).backward()
                loss_avg += float(loss.detach()) * accum

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
            # Cosine schedule only after stable period for non-v3 groups
            if self.step >= self.stable_steps:
                self.sched.step()
            self.step += 1

            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))

            if self.step % log_every == 0 or self.step == 1:
                lr = self.opt.param_groups[0]['lr']
                lr_map = {g.get('name', str(i)): float(g['lr'])
                          for i, g in enumerate(self.opt.param_groups)}
                row = dict(
                    step=self.step, loss=loss_avg / float(accum),
                    lr=lr, lr_groups=lr_map, gnorm=float(gnorm),
                    used_ref=bool(use_ref),
                    batch_kind=batch_kind,
                    teacher_l1=teacher_l,
                    loss_ref=loss_ref_v,
                    loss_noref=loss_noref_v,
                    loss_pure_noref=loss_pure_noref_v,
                    lambda_no_ref=self.lambda_no_ref if self.dual_forward else 0.0,
                    no_ref_prob=self.no_ref_prob,
                    sample_stats=dict(sample_stats),
                    stable=bool(self.step <= self.stable_steps),
                    peak_mem_mb=peak_mem, clip_events=clipped,
                    accum=accum, effective_batch=eff_batch,
                )
                aux = getattr(self.model, 'last_aux', None) or {}
                if aux.get('c_t_stats'):
                    row['c_t'] = aux['c_t_stats']
                    row['c_l'] = aux['c_l_stats']
                if aux.get('match'):
                    row['match'] = aux['match']
                if aux.get('v3') and aux['v3'].get('inject_res_norm'):
                    row['inject_res_norm'] = aux['v3']['inject_res_norm']
                if aux.get('v4') and aux['v4'].get('tex_res_norm'):
                    row['tex_res_norm'] = aux['v4']['tex_res_norm']
                self.log_f.write(json.dumps(row) + '\n')
                self.log_f.flush()
                print('step %d/%d loss=%.4f lr=%.2e ref=%s accum=%d mem=%.0fMB' % (
                    self.step, max_steps, row['loss'], lr, use_ref, accum, peak_mem),
                    flush=True)

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
            sample_stats=dict(sample_stats),
        )
        with open(os.path.join(self.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        self.log_f.close()
        print('=== done in %.1f min ===' % (summary['seconds'] / 60), flush=True)
        return summary
