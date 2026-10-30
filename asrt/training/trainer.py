"""ASRT training loop with true gradient accumulation, AMP, clip, resume."""
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
from asrt.evaluation.protocol import infer_with_protocol, resolve_eval_cfg
from asrt.models.asrt import ASRT, build_asrt, load_pretrained_into_v3
from asrt.models.asrt_v4 import ASRTv4, build_asrt_v4, load_stage_a_into_v4
from asrt.training.optimizer import build_optimizer
from asrt.training.scheduler import build_scheduler, preview_lrs
from asrt.training.seed_utils import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


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


def _count_domains(batch) -> dict:
    out = dict(real=0, syn=0, other=0)
    domains = batch.get('domain')
    if domains is None:
        # fallback: count by batch size as other
        bs = int(batch['low'].size(0)) if 'low' in batch else 1
        out['other'] = bs
        return out
    if torch.is_tensor(domains):
        domains = [str(d) for d in domains]
    elif isinstance(domains, str):
        domains = [domains]
    for d in domains:
        dl = str(d).lower()
        if dl == 'real':
            out['real'] += 1
        elif dl in ('syn', 'synthetic'):
            out['syn'] += 1
        else:
            out['other'] += 1
    return out


class Trainer:
    def __init__(self, cfg: dict, device: Optional[str] = None):
        self.cfg = dict(cfg)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(cfg.get('seed', 42))
        seed_everything(self.seed)
        self.dl_generator = make_generator(self.seed)

        model_cfg = dict(cfg.get('model', {}) or {})
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
        self.optimizer_steps = 0
        self.amp_skips = 0
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
        self.eval_cfg = resolve_eval_cfg(cfg)
        self.teacher = None
        t_ckpt = cfg.get('teacher_ckpt') or ''
        if (not self.is_v4) and self.teacher_w > 0 and t_ckpt and os.path.isfile(t_ckpt):
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

        # LR schedule preview at construction
        try:
            prev = preview_lrs(self.opt, cfg)
            print('lr_schedule_preview %s' % json.dumps(prev), flush=True)
        except Exception as e:
            print('lr_schedule_preview failed: %s' % e, flush=True)

    def _apply_stable_lrs(self):
        """During stable_steps: keep v3_new/v4_new LR, zero other group LRs."""
        if self.stable_steps <= 0:
            return
        if self.step >= self.stable_steps:
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
                rows, crop=crop, seed=self.seed,
                augment=True, allow_missing_ref=False, query_align=qa)
        else:
            ds = PairedRestoreDataset(
                rows, crop=crop, seed=self.seed, augment=True)
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
            drop_last=True, collate_fn=collate_skip_none_ref,
            worker_init_fn=seed_worker if workers > 0 else None,
            generator=self.dl_generator if shuffle else None,
        )

    def save(self, tag: str, extra: Optional[dict] = None):
        path = os.path.join(self.out_dir, 'checkpoints', '%s.pt' % tag)
        payload = dict(
            model=self.model.state_dict(),
            optimizer=self.opt.state_dict(),
            scheduler=self.sched.state_dict(),
            scaler=self.scaler.state_dict(),
            step=self.step,
            optimizer_steps=self.optimizer_steps,
            amp_skips=self.amp_skips,
            cfg=self.cfg,
            rng=capture_rng_state(),
            # legacy keys kept for older readers
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            dl_generator=self.dl_generator.get_state(),
            best=self.best,
            eval_cfg=self.eval_cfg,
            resume_note=(
                'Resume restores model/optimizer/scheduler/RNG/DataLoader generator '
                'but does not guarantee bitwise-identical data ordering after an '
                'interruption (iterator cursor is not serialized).'
            ),
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
            self.optimizer_steps = int(ck.get('optimizer_steps', self.step))
            self.amp_skips = int(ck.get('amp_skips', 0))
            self.best = ck.get('best')
            if 'optimizer' in ck:
                self.opt.load_state_dict(ck['optimizer'])
            if 'scheduler' in ck:
                self.sched.load_state_dict(ck['scheduler'])
            if 'scaler' in ck:
                self.scaler.load_state_dict(ck['scaler'])
            if ck.get('rng') is not None:
                restore_rng_state(ck['rng'])
            elif ck.get('rng_torch') is not None:
                torch.set_rng_state(ck['rng_torch'])
                if ck.get('rng_cuda') is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(ck['rng_cuda'])
            if ck.get('dl_generator') is not None:
                self.dl_generator.set_state(ck['dl_generator'])
            print(
                'resume note: model/optim/sched/RNG restored; DataLoader iterator '
                'cursor is NOT guaranteed identical after interruption.',
                flush=True)
        else:
            self.step = 0
            self.optimizer_steps = 0
            self.amp_skips = 0
            self.best = None
        print('loaded %s @ step %d (resume_optim=%s)' % (
            path, self.step, resume_optim), flush=True)

    @torch.no_grad()
    def evaluate(self, rows, with_ref: bool, limit: int = 0):
        from asrt.data.paired import read_rgb_m11

        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scores = []
        sample_ids = []
        n = len(rows) if not limit else min(limit, len(rows))
        skipped_missing_ref = 0
        for i in range(n):
            row = rows[i]
            low = read_rgb_m11(row['low_path'])[None].to(self.device)
            gt = read_rgb_m11(row['gt_path'])[None].to(self.device)
            if tuple(gt.shape[-2:]) != tuple(low.shape[-2:]):
                gt = F.interpolate(gt, size=low.shape[-2:], mode='bilinear', align_corners=False)
            ref = None
            if with_ref:
                if not row.get('ref_path'):
                    skipped_missing_ref += 1
                    continue  # never silent-fallback to no-ref in reference mode
                ref = read_rgb_m11(row['ref_path'])[None].to(self.device)
                if tuple(ref.shape[-2:]) != tuple(low.shape[-2:]):
                    ref = F.interpolate(ref, size=low.shape[-2:], mode='bilinear', align_corners=False)
            pred, meta = infer_with_protocol(
                self.model, low, ref, eval_cfg=self.cfg)
            scores.append(psnr_ssim_rgb(
                pred, gt, input_range=self.eval_cfg.get('input_range', 'minus_one_one')))
            sample_ids.append(row.get('sample_id') or row.get('id') or row.get('low_path'))
            del low, gt, ref, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not scores:
            return dict(
                psnr_rgb=0.0, ssim_rgb=0.0, n=0,
                skipped_missing_ref=skipped_missing_ref,
                eval_cfg=self.eval_cfg, sample_ids=[])
        return dict(
            psnr_rgb=float(np.mean([s['psnr_rgb'] for s in scores])),
            ssim_rgb=float(np.mean([s['ssim_rgb'] for s in scores])),
            n=len(scores),
            skipped_missing_ref=skipped_missing_ref,
            eval_cfg=dict(self.eval_cfg),
            sample_ids=sample_ids,
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
        print('train accum=%d batch=%d effective_batch=%d (true micro-batch loop)' % (
            accum, micro_bs, eff_batch), flush=True)
        print(
            'sampling: no_ref_prob=%.3f dual_forward=%s lambda_no_ref=%.3f'
            % (self.no_ref_prob, self.dual_forward, self.lambda_no_ref),
            flush=True)
        print('eval_protocol %s' % json.dumps(self.eval_cfg), flush=True)

        t0 = time.time()
        peak_mem = 0.0
        clipped = 0
        # Cumulative across run
        cum = dict(
            ref_micro_batches=0, pure_noref_micro_batches=0,
            dual_noref_loss_sum=0.0, pure_noref_loss_sum=0.0, loss_ref_sum=0.0,
            dual_noref_n=0, pure_noref_n=0, loss_ref_n=0,
            domain_real=0, domain_syn=0, domain_other=0,
            samples_processed=0, micro_batches=0,
        )
        self.log_f = open(os.path.join(self.out_dir, 'train.jsonl'), 'a')
        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        while self.step < max_steps:
            self.opt.zero_grad(set_to_none=True)
            self._apply_stable_lrs()

            # Per-step aggregates (only over micros that actually ran)
            step_loss_sum = 0.0
            step_n_micro = 0
            step_ref_n = 0
            step_pure_n = 0
            step_loss_ref_sum = 0.0
            step_loss_dual_noref_sum = 0.0
            step_loss_pure_sum = 0.0
            step_samples = 0
            step_dom = dict(real=0, syn=0, other=0)
            last_aux = None
            teacher_l = 0.0

            for _micro in range(accum):
                # --- sample micro-batch ---
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

                bs = int(low.size(0))
                step_samples += bs
                cum['samples_processed'] += bs
                cum['micro_batches'] += 1
                step_n_micro += 1
                dom = _count_domains(batch)
                step_dom['real'] += dom['real']
                step_dom['syn'] += dom['syn']
                step_dom['other'] += dom['other']
                cum['domain_real'] += dom['real']
                cum['domain_syn'] += dom['syn']
                cum['domain_other'] += dom['other']

                with autocast(enabled=self.scaler.is_enabled()):
                    if self.dual_forward and use_ref and ref is not None:
                        pred_ref = self.model(low, ref=ref, query_gamma=qg, query_lut=ql)
                        aux_ref = getattr(self.model, 'last_aux', None)
                        pred_noref = self.model(low, ref=None)
                        if aux_ref is not None:
                            self.model.last_aux = aux_ref
                            last_aux = aux_ref
                        loss_ref = l1_loss(pred_ref, gt)
                        loss_noref = l1_loss(pred_noref, gt)
                        raw_loss = loss_ref + self.lambda_no_ref * loss_noref
                        lr_v = float(loss_ref.detach())
                        ln_v = float(loss_noref.detach())
                        step_ref_n += 1
                        step_loss_ref_sum += lr_v
                        step_loss_dual_noref_sum += ln_v
                        cum['ref_micro_batches'] += 1
                        cum['loss_ref_sum'] += lr_v
                        cum['loss_ref_n'] += 1
                        cum['dual_noref_loss_sum'] += ln_v
                        cum['dual_noref_n'] += 1
                    else:
                        pred = self.model(low, ref=ref, query_gamma=qg, query_lut=ql)
                        raw_loss = l1_loss(pred, gt)
                        if (self.teacher is not None and self.teacher_w > 0
                                and (not use_ref or ref is None)):
                            with torch.no_grad():
                                t_pred = self.teacher(low, ref=None)
                            t_loss = l1_loss(pred, t_pred)
                            raw_loss = raw_loss + self.teacher_w * t_loss
                            teacher_l = float(t_loss.detach())
                        lp_v = float(raw_loss.detach())
                        step_pure_n += 1
                        step_loss_pure_sum += lp_v
                        cum['pure_noref_micro_batches'] += 1
                        cum['pure_noref_loss_sum'] += lp_v
                        cum['pure_noref_n'] += 1
                        last_aux = getattr(self.model, 'last_aux', None)

                    loss = raw_loss / float(accum)
                    if not torch.isfinite(loss):
                        raise SystemExit('non-finite loss at step %d micro' % self.step)
                    self.scaler.scale(loss).backward()
                    step_loss_sum += float(raw_loss.detach())

            # --- optimizer step (once per global step) ---
            self.scaler.unscale_(self.opt)
            if self.clip > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, self.clip)
                if float(gnorm) > self.clip:
                    clipped += 1
            else:
                gnorm = 0.0

            scale_before = float(self.scaler.get_scale())
            self.scaler.step(self.opt)
            self.scaler.update()
            scale_after = float(self.scaler.get_scale())
            amp_skipped = scale_after < scale_before
            if amp_skipped:
                self.amp_skips += 1
            else:
                self.optimizer_steps += 1
                # Scheduler only when optimizer actually stepped; and after stable
                if self.step >= self.stable_steps:
                    self.sched.step()

            self.step += 1

            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))

            if self.step % log_every == 0 or self.step == 1:
                lr_map = {g.get('name', str(i)): float(g['lr'])
                          for i, g in enumerate(self.opt.param_groups)}
                loss_total = step_loss_sum / max(step_n_micro, 1)
                row = dict(
                    step=self.step,
                    optimizer_steps=self.optimizer_steps,
                    micro_batches=step_n_micro,
                    samples_processed=step_samples,
                    effective_batch=eff_batch,
                    loss_total=loss_total,
                    loss_ref_mean=(step_loss_ref_sum / step_ref_n) if step_ref_n else None,
                    loss_dual_noref_mean=(
                        step_loss_dual_noref_sum / step_ref_n) if step_ref_n else None,
                    loss_pure_noref_mean=(
                        step_loss_pure_sum / step_pure_n) if step_pure_n else None,
                    ref_micro_batches=step_ref_n,
                    pure_noref_micro_batches=step_pure_n,
                    domain_real_samples=step_dom['real'],
                    domain_syn_samples=step_dom['syn'],
                    domain_other_samples=step_dom['other'],
                    grad_norm=float(gnorm),
                    lr_groups=lr_map,
                    amp_step_skipped=bool(amp_skipped),
                    amp_skips_total=self.amp_skips,
                    teacher_l1=teacher_l,
                    lambda_no_ref=self.lambda_no_ref if self.dual_forward else 0.0,
                    no_ref_prob=self.no_ref_prob,
                    sample_stats_cum=dict(cum),
                    stable=bool(self.step <= self.stable_steps),
                    peak_mem_mb=peak_mem,
                    clip_events=clipped,
                    accum=accum,
                )
                aux = last_aux or {}
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
                print(
                    'step %d/%d loss=%.4f opt_steps=%d micros=%d samples=%d '
                    'ref_m=%d pure_m=%d amp_skip=%s mem=%.0fMB'
                    % (self.step, max_steps, loss_total, self.optimizer_steps,
                       step_n_micro, step_samples, step_ref_n, step_pure_n,
                       amp_skipped, peak_mem),
                    flush=True)

            if eval_rows and (self.step % eval_every == 0 or self.step == max_steps):
                metrics = self.evaluate(
                    eval_rows,
                    with_ref=use_ref_default and any(r.get('ref_path') for r in eval_rows),
                    limit=int(self.cfg.get('eval_limit', 0)))
                print('  eval PSNR=%.4f SSIM=%.4f n=%d mode=%s skipped_noref=%d' % (
                    metrics['psnr_rgb'], metrics['ssim_rgb'], metrics['n'],
                    metrics.get('eval_cfg', {}).get('inference_mode'),
                    metrics.get('skipped_missing_ref', 0)), flush=True)
                if self.best is None or metrics['psnr_rgb'] > self.best['psnr_rgb']:
                    self.best = dict(step=self.step, **{
                        k: metrics[k] for k in (
                            'psnr_rgb', 'ssim_rgb', 'n', 'eval_cfg',
                            'skipped_missing_ref') if k in metrics
                    })
                    self.save('best', extra=metrics)
                self.model.train()

            if self.step % save_every == 0:
                self.save('step_%06d' % self.step)

        self.save('last')
        summary = dict(
            steps=self.step,
            optimizer_steps=self.optimizer_steps,
            amp_skips=self.amp_skips,
            seconds=time.time() - t0,
            peak_mem_mb=peak_mem,
            clip_events=clipped,
            best=self.best,
            cfg=self.cfg,
            eval_cfg=self.eval_cfg,
            sample_stats=dict(cum),
        )
        with open(os.path.join(self.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        self.log_f.close()
        print('=== done in %.1f min ===' % (summary['seconds'] / 60), flush=True)
        return summary
