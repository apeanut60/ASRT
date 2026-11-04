"""Formal IRP v1 trainer: Adam, warmup+cosine, AMP, accum, resume, val."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from asrt.data.halo_crop import center_crop, compute_irp_halo
from asrt.data.irp_dataset import (
    TripletIRPDataset,
    assert_no_overlap,
    collate_irp,
    manifest_hash,
)
from asrt.data.manifest import load_manifest
from asrt.data.reference_modes import build_wrong_mapping
from asrt.evaluation.metrics import psnr_ssim_rgb
from asrt.evaluation.protocol import infer_with_protocol, resolve_eval_cfg
from asrt.models.asrt_irp import build_asrt_irp, extract_irp_hparams
from asrt.models.factory import is_irp_cfg
from asrt.training.seed_utils import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def validate_irp_cfg(cfg: dict, rows_train: List[dict], rows_val: Optional[List[dict]] = None):
    """P2 protocol asserts for IRP training."""
    if not is_irp_cfg(cfg):
        raise ValueError('config is not IRP')
    model = dict(cfg.get('model') or {})
    tr = dict(cfg.get('training') or {})
    attn = dict(model.get('attention') or {})
    prior_source = str(model.get('prior_source', 'nano')).lower()
    n_feats = int(model.get('n_feats', 64))
    heads = int(attn.get('heads', 4))
    ws = int(attn.get('window_size', 8))
    if ws < 1:
        raise ValueError('illegal window_size')
    if n_feats % heads != 0:
        raise ValueError('attention heads must divide n_feats')
    if not rows_train:
        raise ValueError('training manifest is empty')
    bs = int(tr.get('batch_size', 8))
    if bs > len(rows_train) and bool(tr.get('drop_last', True)):
        raise ValueError(
            'batch_size=%d > n_train=%d with drop_last → no batches'
            % (bs, len(rows_train)))
    amp = bool(tr.get('amp', cfg.get('amp', True)))
    device = str(tr.get('device', cfg.get('device', 'cuda')))
    if amp and not str(device).startswith('cuda'):
        raise ValueError(
            'AMP requested but device=%r is not CUDA; refusing silent fake-AMP'
            % device)
    if prior_source == 'nano':
        missing = [r.get('sample_id') for r in rows_train if not r.get('ref_path')]
        if missing:
            raise RuntimeError(
                'prior_source=nano training requires ref for all samples; '
                'missing e.g. %s' % missing[:5])
    if rows_val is not None:
        assert_no_overlap(rows_train, rows_val)
    return True


class WarmupCosineScheduler:
    """Linear warmup then cosine to eta_min (absolute)."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int, eta_min: float):
        self.optimizer = optimizer
        self.total = max(int(total_steps), 1)
        self.warmup = max(int(warmup_steps), 0)
        self.eta_min = float(eta_min)
        self.last_epoch = -1
        self.base_lrs = [float(g['lr']) for g in optimizer.param_groups]

    def _lr_at(self, step: int) -> list:
        t = max(0, int(step))
        out = []
        for base in self.base_lrs:
            if self.warmup > 0 and t < self.warmup:
                out.append(base * float(t + 1) / float(self.warmup))
            else:
                denom = max(1, self.total - self.warmup)
                prog = min(1.0, float(max(0, t - self.warmup)) / float(denom))
                cos = 0.5 * (1.0 + math.cos(math.pi * prog))
                out.append(self.eta_min + (base - self.eta_min) * cos)
        return out

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)
        lrs = self._lr_at(self.last_epoch)
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g['lr'] = lr

    def state_dict(self):
        return dict(
            last_epoch=self.last_epoch, total=self.total, warmup=self.warmup,
            eta_min=self.eta_min, base_lrs=list(self.base_lrs))

    def load_state_dict(self, state):
        self.last_epoch = int(state.get('last_epoch', -1))
        self.total = int(state.get('total', self.total))
        self.warmup = int(state.get('warmup', self.warmup))
        self.eta_min = float(state.get('eta_min', self.eta_min))
        if state.get('base_lrs') is not None:
            self.base_lrs = [float(x) for x in state['base_lrs']]
        if self.last_epoch >= 0:
            lrs = self._lr_at(self.last_epoch)
            for g, lr in zip(self.optimizer.param_groups, lrs):
                g['lr'] = lr

    def get_last_lr(self):
        return [float(g['lr']) for g in self.optimizer.param_groups]


def _load_rows(cfg: dict, split: str) -> List[dict]:
    data = dict(cfg.get('data') or {})
    man_root = Path(data.get('manifest_root', '/root/data/experiments/ASRT/manifests'))
    key = 'train_manifest' if split == 'train' else 'val_manifest'
    default = ('lolv2_real_triplet_train.json' if split == 'train'
               else 'lolv2_real_triplet_dev.json')
    path = Path(data.get(key, man_root / default))
    if not path.is_absolute():
        path = man_root / path
    if not path.exists():
        raise FileNotFoundError('manifest not found: %s' % path)
    rows = load_manifest(str(path))
    # Attach nano ref if missing
    ref_root = Path(data.get('real_ref', ''))
    if ref_root and ref_root.exists():
        for r in rows:
            if not r.get('ref_path'):
                cand = ref_root / (str(r['sample_id']) + '.png')
                if not cand.exists():
                    # LOLv2 refs sometimes match low filename
                    low = Path(r.get('low_path', ''))
                    cand2 = ref_root / low.name
                    cand = cand2 if cand2.exists() else cand
                if cand.exists():
                    r['ref_path'] = str(cand)
    return list(rows)


class IRPTrainer:
    def __init__(self, cfg: dict, device: Optional[str] = None):
        self.cfg = dict(cfg)
        tr = dict(cfg.get('training') or {})
        self.tr = tr
        self.device = device or str(tr.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.seed = int(tr.get('seed', cfg.get('seed', 42)))
        seed_everything(self.seed)
        self.dl_generator = make_generator(self.seed)

        self.model = build_asrt_irp(cfg).to(self.device)
        self.prior_source = str(self.model.prior_source)
        opt_name = str(tr.get('optimizer', 'adam')).lower()
        lr = float(tr.get('learning_rate', tr.get('lr', 1e-4)))
        wd = float(tr.get('weight_decay', 1e-4))
        betas = tuple(tr.get('betas', (0.9, 0.999)))
        params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name in ('adamw',):
            self.opt = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
        else:
            self.opt = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)

        self.max_steps = int(tr.get('max_steps') or 0)
        if self.max_steps <= 0:
            raise ValueError('training.max_steps must be > 0 for formal IRPTrainer')
        warmup = int(tr.get('warmup_steps', max(100, self.max_steps // 50)))
        eta_min = float(tr.get('eta_min', 1e-6))
        self.sched = WarmupCosineScheduler(
            self.opt, total_steps=self.max_steps, warmup_steps=warmup, eta_min=eta_min)
        self.amp = bool(tr.get('amp', True)) and str(self.device).startswith('cuda')
        if bool(tr.get('amp', True)) and not self.amp:
            raise ValueError('AMP requested but CUDA unavailable')
        self.scaler = GradScaler(enabled=self.amp)
        self.clip = float(tr.get('grad_clip', 1.0))
        self.accum = max(1, int(tr.get('grad_accum', tr.get('accumulate', 1))))
        self.crop = int(tr.get('crop_size', 128))
        self.batch_size = int(tr.get('batch_size', 8))
        sigma = float(dict(cfg.get('model') or {}).get('reference_prior', {}).get(
            'gaussian_sigma', 7.0))
        _halo = tr.get('halo', None)
        if _halo is None:
            self.halo = int(compute_irp_halo(
                sigma=sigma, window_size=int(self.model.window_size)))
        else:
            self.halo = int(_halo)

        self.out_dir = Path(cfg.get('out_dir', 'outputs/irp_v1'))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'checkpoints').mkdir(exist_ok=True)
        self.log_path = self.out_dir / 'train_log.jsonl'
        self.step = 0
        self.optimizer_steps = 0
        self.epoch = 0
        self.best = None
        self.best_psnr = None
        self.micro_in_accum = 0

        self.rows_train = _load_rows(cfg, 'train')
        self.rows_val = _load_rows(cfg, 'val')
        validate_irp_cfg(cfg, self.rows_train, self.rows_val)
        self.train_hash = manifest_hash(self.rows_train)
        self.val_hash = manifest_hash(self.rows_val)

        # Wrong mapping for diagnostics (not used in Correct training)
        wmap_path = dict(cfg.get('data') or {}).get('wrong_mapping_path')
        if wmap_path:
            self.wrong_mapping_path = str(wmap_path)
        else:
            self.wrong_mapping_path = str(self.out_dir / 'wrong_ref_mapping.json')
            if not os.path.exists(self.wrong_mapping_path):
                build_wrong_mapping(
                    [r['sample_id'] for r in self.rows_train + self.rows_val],
                    seed=self.seed, out_path=self.wrong_mapping_path)

        require_ref = self.prior_source == 'nano'
        self.ds_train = TripletIRPDataset(
            self.rows_train, crop=self.crop, seed=self.seed, augment=True,
            epoch=0, halo=self.halo, window_size=self.model.window_size,
            sigma=sigma, return_halo=True, require_ref=require_ref,
            ref_mode=str(dict(cfg.get('data') or {}).get('ref_mode', 'correct')),
            id_to_row={str(r['sample_id']): r for r in self.rows_train + self.rows_val},
        )
        self.ds_val = TripletIRPDataset(
            self.rows_val, crop=0, seed=self.seed, augment=False,
            halo=0, require_ref=require_ref, ref_mode='correct',
            id_to_row={str(r['sample_id']): r for r in self.rows_val},
        )
        self.loader = self._make_loader()
        self.loader_iter = None
        self.samples_seen_in_epoch = 0
        self.eval_cfg = resolve_eval_cfg(cfg)

        with open(self.out_dir / 'resolved_config.json', 'w') as f:
            json.dump(cfg, f, indent=2)

    def _make_loader(self) -> DataLoader:
        nw = int(self.tr.get('num_workers', 2))
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=nw,
            drop_last=bool(self.tr.get('drop_last', True)),
            worker_init_fn=seed_worker if nw > 0 else None,
            generator=self.dl_generator,
            collate_fn=collate_irp,
            pin_memory=str(self.device).startswith('cuda'),
        )

    def _next_batch(self):
        if self.loader_iter is None:
            self.loader_iter = iter(self.loader)
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.epoch += 1
            self.ds_train.set_epoch(self.epoch)
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)
            self.samples_seen_in_epoch = 0
        self.samples_seen_in_epoch += int(batch['low'].size(0))
        return batch

    def _forward_loss(self, batch) -> Tuple[torch.Tensor, dict]:
        low = batch['low'].to(self.device, non_blocking=True)
        gt = batch['gt'].to(self.device, non_blocking=True)
        ref = batch['ref']
        if ref is not None:
            ref = ref.to(self.device, non_blocking=True)
        halo = int(batch.get('halo', 0) or 0)
        crop = int(batch.get('crop', self.crop) or self.crop)
        pred = self.model(low, ref)
        if halo > 0:
            pred_c = center_crop(pred, halo, crop)
            gt_c = center_crop(gt, halo, crop)
        else:
            pred_c, gt_c = pred, gt
        loss = F.mse_loss(pred_c, gt_c)
        aux = dict(
            sample_ids=batch.get('sample_id'),
            pred_min=float(pred.detach().min().item()),
            pred_max=float(pred.detach().max().item()),
            pred_mean=float(pred.detach().mean().item()),
        )
        return loss, aux

    def _grad_norms(self) -> dict:
        total = 0.0
        ref_branch = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = float(p.grad.data.norm().item())
            total += g * g
            if 'attn_ref' in name or 'ref_projection' in name or 'ref_prior' in name:
                ref_branch += g * g
        return dict(
            gradient_norm=math.sqrt(total),
            reference_branch_grad_norm=math.sqrt(ref_branch),
        )

    def _alpha_beta(self) -> dict:
        bb = self.model.backbone
        out = {}
        for tag, blk in (('h4', bb.coarse_irp), ('h2', bb.middle_irp), ('h1', bb.fine_irp)):
            out['alpha_%s' % tag] = float(blk.alpha.detach().item())
            if blk.beta is not None:
                out['beta_%s' % tag] = float(blk.beta.detach().item())
        return out

    def train_micro_step(self) -> dict:
        self.model.train()
        batch = self._next_batch()
        with autocast(enabled=self.amp):
            loss, aux = self._forward_loss(batch)
            loss_scaled = loss / float(self.accum)
        self.scaler.scale(loss_scaled).backward()
        self.micro_in_accum += 1
        self.step += 1
        did_step = False
        grad_info = {}
        if self.micro_in_accum >= self.accum:
            self.scaler.unscale_(self.opt)
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            grad_info = self._grad_norms()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            self.optimizer_steps += 1
            self.micro_in_accum = 0
            did_step = True
        peak = 0.0
        if str(self.device).startswith('cuda'):
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        log = dict(
            step=self.step,
            optimizer_steps=self.optimizer_steps,
            epoch=self.epoch,
            loss=float(loss.item()),
            learning_rate=float(self.opt.param_groups[0]['lr']),
            grad_scaler=float(self.scaler.get_scale()) if self.amp else 1.0,
            peak_memory=peak,
            prior_source=self.prior_source,
            did_optimizer_step=did_step,
            sample_ids=aux.get('sample_ids'),
            **self._alpha_beta(),
            **grad_info,
            pred_min=aux['pred_min'], pred_max=aux['pred_max'], pred_mean=aux['pred_mean'],
        )
        return log

    @torch.no_grad()
    def validate(self, max_images: Optional[int] = None) -> dict:
        self.model.eval()
        n = 0
        psnrs, ssims = [], []
        limit = max_images or int(self.tr.get('val_max_images', len(self.ds_val)))
        for i in range(min(len(self.ds_val), limit)):
            item = self.ds_val[i]
            low = item['low'][None].to(self.device)
            gt = item['gt'][None].to(self.device)
            ref = item['ref']
            if ref is not None:
                ref = ref[None].to(self.device)
            pred, meta = infer_with_protocol(
                self.model, low, ref, eval_cfg=self.cfg)
            m = psnr_ssim_rgb(pred, gt, input_range='minus_one_one')
            psnrs.append(float(m['psnr_rgb']))
            ssims.append(float(m['ssim_rgb']))
            n += 1
        if not psnrs:
            return dict(n=0, psnr=0.0, ssim=0.0)
        return dict(
            n=n,
            psnr=sum(psnrs) / len(psnrs),
            ssim=sum(ssims) / len(ssims),
            inference_used=meta.get('inference_used'),
        )

    def state_dict_ckpt(self) -> dict:
        return dict(
            model=self.model.state_dict(),
            optimizer=self.opt.state_dict(),
            scheduler=self.sched.state_dict(),
            scaler=self.scaler.state_dict(),
            step=self.step,
            optimizer_steps=self.optimizer_steps,
            epoch=self.epoch,
            micro_in_accum=self.micro_in_accum,
            samples_seen_in_epoch=self.samples_seen_in_epoch,
            best=self.best,
            best_psnr=self.best_psnr,
            cfg=self.cfg,
            model_family='asrt_irp',
            model_version='irp_v1',
            prior_source=self.prior_source,
            irp_hparams=extract_irp_hparams(self.cfg),
            train_manifest_hash=self.train_hash,
            val_manifest_hash=self.val_hash,
            seed=self.seed,
            rng=capture_rng_state(),
            dl_generator_state=self.dl_generator.get_state(),
            halo=self.halo,
            wrong_mapping_path=self.wrong_mapping_path,
        )

    def save_checkpoint(self, tag: str):
        path = self.out_dir / 'checkpoints' / ('%s.pt' % tag)
        torch.save(self.state_dict_ckpt(), path)
        return str(path)

    def load_checkpoint(self, path: str):
        ck = torch.load(path, map_location='cpu')
        # family / prior_source checks
        if str(ck.get('model_family', '')).lower() not in ('asrt_irp', 'irp', ''):
            # allow if irp_hparams present
            if not isinstance(ck.get('irp_hparams'), dict):
                raise RuntimeError('checkpoint is not IRP')
        ck_ps = str(ck.get('prior_source', ck.get('irp_hparams', {}).get('prior_source', '')))
        if ck_ps and ck_ps != self.prior_source:
            raise RuntimeError(
                'prior_source mismatch ckpt=%s model=%s' % (ck_ps, self.prior_source))
        self.model.load_state_dict(ck['model'], strict=True)
        self.opt.load_state_dict(ck['optimizer'])
        self.sched.load_state_dict(ck['scheduler'])
        if 'scaler' in ck:
            self.scaler.load_state_dict(ck['scaler'])
        self.step = int(ck.get('step', 0))
        self.optimizer_steps = int(ck.get('optimizer_steps', 0))
        self.epoch = int(ck.get('epoch', 0))
        self.micro_in_accum = int(ck.get('micro_in_accum', 0))
        self.samples_seen_in_epoch = int(ck.get('samples_seen_in_epoch', 0))
        self.best = ck.get('best')
        self.best_psnr = ck.get('best_psnr')
        self.ds_train.set_epoch(self.epoch)
        if ck.get('dl_generator_state') is not None:
            self.dl_generator.set_state(ck['dl_generator_state'])
        restore_rng_state(ck.get('rng'))
        # rebuild loader so shuffle continues from generator state
        self.loader = self._make_loader()
        self.loader_iter = None
        return ck

    def fit(self):
        self.opt.zero_grad(set_to_none=True)
        val_every = int(self.tr.get('val_every', 1000))
        ckpt_every = int(self.tr.get('ckpt_every', 2000))
        log_every = int(self.tr.get('log_every', 50))
        while self.step < self.max_steps:
            log = self.train_micro_step()
            if self.step % log_every == 0 or self.step == 1:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(log) + '\n')
                print('step', self.step, 'loss', log['loss'], 'lr', log['learning_rate'],
                      flush=True)
            if val_every > 0 and self.optimizer_steps > 0 and (
                    self.optimizer_steps % val_every == 0) and log.get('did_optimizer_step'):
                metrics = self.validate()
                metrics['step'] = self.step
                with open(self.out_dir / 'val_log.jsonl', 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                if self.best_psnr is None or metrics['psnr'] > self.best_psnr:
                    self.best_psnr = metrics['psnr']
                    self.best = metrics
                    self.save_checkpoint('best_validation_psnr')
            if ckpt_every > 0 and self.step % ckpt_every == 0:
                self.save_checkpoint('periodic_%d' % self.step)
        self.save_checkpoint('last')
        return dict(best=self.best, best_psnr=self.best_psnr, step=self.step)
