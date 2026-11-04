"""Formal IRP v1 trainer (train protocol v2).

Step semantics:
  global_step / max_steps = Optimizer Update count
  1 update = grad_accum micro-batches + 1 optimizer.step (if not AMP-skipped)

Checkpoints are written only on update boundaries (no partial accum state).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from asrt.models.factory import compare_irp_hparams, is_irp_cfg
from asrt.training.resumable_sampler import ResumableBatchSampler
from asrt.training.seed_utils import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    seed_worker,
)

PROTOCOL_VERSION = 'irp_v1_train_v2'


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def git_meta(repo: Optional[str] = None) -> dict:
    root = repo or str(Path(__file__).resolve().parents[2])
    meta = dict(git_commit=None, git_dirty=None)
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
        meta['git_commit'] = commit
        meta['git_dirty'] = bool(dirty)
    except Exception:
        meta['git_commit'] = 'unknown'
        meta['git_dirty'] = True
    return meta


def validate_irp_cfg(cfg: dict, rows_train: List[dict], rows_val: Optional[List[dict]] = None):
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


def extract_train_protocol(cfg: dict, *, train_hash: str = '', val_hash: str = '',
                           dataset_size: int = 0, halo: int = 0) -> dict:
    tr = dict(cfg.get('training') or {})
    accum = max(1, int(tr.get('grad_accum', tr.get('accumulate', 1))))
    bs = int(tr.get('batch_size', 8))
    opt = str(tr.get('optimizer', 'adam')).lower()
    return dict(
        protocol_version=PROTOCOL_VERSION,
        optimizer=opt,
        base_lr=float(tr.get('learning_rate', tr.get('lr', 1e-4))),
        weight_decay=float(tr.get('weight_decay', 1e-4)),
        betas=list(tr.get('betas', (0.9, 0.999))),
        micro_batch_size=bs,
        grad_accum=accum,
        effective_batch_size=bs * accum,
        crop_size=int(tr.get('crop_size', 128)),
        halo=int(halo),
        loss=str(tr.get('loss', 'mse')),
        scheduler='warmup_cosine',
        warmup_steps=int(tr.get('warmup_steps', 0)),
        max_steps=int(tr.get('max_steps') or 0),
        eta_min=float(tr.get('eta_min', 1e-6)),
        amp=bool(tr.get('amp', True)),
        grad_clip=float(tr.get('grad_clip', 1.0)),
        drop_last=bool(tr.get('drop_last', True)),
        seed=int(tr.get('seed', cfg.get('seed', 42))),
        train_manifest_hash=str(train_hash),
        val_manifest_hash=str(val_hash),
        dataset_size=int(dataset_size),
        prior_source=str(dict(cfg.get('model') or {}).get('prior_source', '')),
    )


def compare_train_protocol(a: dict, b: dict) -> List[dict]:
    keys = (
        'protocol_version', 'optimizer', 'base_lr', 'weight_decay', 'betas',
        'micro_batch_size', 'grad_accum', 'effective_batch_size', 'crop_size',
        'halo', 'loss', 'scheduler', 'warmup_steps', 'max_steps', 'eta_min',
        'amp', 'grad_clip', 'drop_last', 'seed', 'train_manifest_hash',
        'val_manifest_hash', 'dataset_size', 'prior_source',
    )
    conflicts = []
    for k in keys:
        if k not in a or k not in b:
            conflicts.append(dict(key=k, expected=a.get(k), got=b.get(k),
                                  reason='missing'))
            continue
        va, vb = a[k], b[k]
        if isinstance(va, (list, tuple)) or isinstance(vb, (list, tuple)):
            la = [float(x) if isinstance(x, (float, int)) else x for x in list(va)]
            lb = [float(x) if isinstance(x, (float, int)) else x for x in list(vb)]
            if la != lb:
                conflicts.append(dict(key=k, expected=va, got=vb))
        elif isinstance(va, float) or isinstance(vb, float):
            if abs(float(va) - float(vb)) > 1e-12:
                conflicts.append(dict(key=k, expected=va, got=vb))
        elif va != vb:
            conflicts.append(dict(key=k, expected=va, got=vb))
    return conflicts


class WarmupCosineScheduler:
    """LR schedule indexed by optimizer update index (0-based).

    Warmup (update_index = 0 .. warmup-1):
        lr = base * (update_index + 1) / warmup
    Cosine (update_index = warmup .. total-1):
        prog = (update_index - warmup) / max(1, total - warmup)
        lr = eta_min + (base - eta_min) * 0.5 * (1 + cos(pi * prog))
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int, eta_min: float):
        self.optimizer = optimizer
        self.total = max(int(total_steps), 1)
        self.warmup = max(int(warmup_steps), 0)
        self.eta_min = float(eta_min)
        self.base_lrs = [float(g['lr']) for g in optimizer.param_groups]
        self.updates_done = 0
        # Apply LR for the first upcoming update immediately.
        self.apply_lr(0)

    def lr_for_update(self, update_index: int) -> List[float]:
        t = max(0, int(update_index))
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

    def apply_lr(self, update_index: int) -> List[float]:
        lrs = self.lr_for_update(update_index)
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g['lr'] = lr
        return lrs

    def mark_update_done(self):
        self.updates_done += 1

    def step(self, epoch=None):
        """Compatibility: treat as mark_update_done + apply next LR."""
        self.mark_update_done()
        self.apply_lr(self.updates_done)

    def state_dict(self):
        return dict(
            updates_done=self.updates_done,
            total=self.total,
            warmup=self.warmup,
            eta_min=self.eta_min,
            base_lrs=list(self.base_lrs),
            # legacy alias
            last_epoch=self.updates_done - 1,
        )

    def load_state_dict(self, state):
        if 'updates_done' in state:
            self.updates_done = int(state['updates_done'])
        elif 'last_epoch' in state:
            # old scheduler: last_epoch was applied update index
            self.updates_done = int(state['last_epoch']) + 1
        self.total = int(state.get('total', self.total))
        self.warmup = int(state.get('warmup', self.warmup))
        self.eta_min = float(state.get('eta_min', self.eta_min))
        if state.get('base_lrs') is not None:
            self.base_lrs = [float(x) for x in state['base_lrs']]
        self.apply_lr(self.updates_done)

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
    ref_root = Path(data.get('real_ref', ''))
    if ref_root and ref_root.exists():
        for r in rows:
            if not r.get('ref_path'):
                cand = ref_root / (str(r['sample_id']) + '.png')
                if not cand.exists():
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
        self.device = device or str(
            tr.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.seed = int(tr.get('seed', cfg.get('seed', 42)))
        seed_everything(self.seed)

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
            raise ValueError('training.max_steps must be > 0 (optimizer update count)')
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
        self.drop_last = bool(tr.get('drop_last', True))
        sigma = float(dict(cfg.get('model') or {}).get('reference_prior', {}).get(
            'gaussian_sigma', 7.0))
        _halo = tr.get('halo', None)
        self.halo = (int(compute_irp_halo(sigma=sigma, window_size=int(self.model.window_size)))
                     if _halo is None else int(_halo))

        self.out_dir = Path(cfg.get('out_dir', 'outputs/irp_v1'))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'checkpoints').mkdir(exist_ok=True)
        self.log_path = self.out_dir / 'train_log.jsonl'

        # Step counters (protocol v2)
        self.global_step = 0       # completed optimizer updates
        self.micro_step = 0        # completed micro-batch backwards
        self.optimizer_updates = 0  # alias of successful updates (== global_step)
        self.skipped_update_count = 0
        self.epoch = 0
        self.batch_cursor = 0
        self.best = None
        self.best_psnr = None
        # legacy alias
        self.step = 0

        self.rows_train = _load_rows(cfg, 'train')
        self.rows_val = _load_rows(cfg, 'val')
        validate_irp_cfg(cfg, self.rows_train, self.rows_val)
        self.train_hash = manifest_hash(self.rows_train)
        self.val_hash = manifest_hash(self.rows_val)
        self.train_protocol = extract_train_protocol(
            cfg, train_hash=self.train_hash, val_hash=self.val_hash,
            dataset_size=len(self.rows_train), halo=self.halo)

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
        self.sampler = ResumableBatchSampler(
            dataset_size=len(self.ds_train),
            batch_size=self.batch_size,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        self.loader = self._make_loader()
        self.loader_iter = None
        self.eval_cfg = resolve_eval_cfg(cfg)
        self.git = git_meta()

        with open(self.out_dir / 'resolved_config.json', 'w') as f:
            json.dump(dict(cfg=cfg, train_protocol=self.train_protocol,
                           git=self.git), f, indent=2)

    def _make_loader(self) -> DataLoader:
        nw = int(self.tr.get('num_workers', 0))
        return DataLoader(
            self.ds_train,
            batch_sampler=self.sampler,
            num_workers=nw,
            worker_init_fn=seed_worker if nw > 0 else None,
            collate_fn=collate_irp,
            pin_memory=str(self.device).startswith('cuda'),
            persistent_workers=False,
        )

    def _sync_epoch_to_dataset(self):
        if self.ds_train.epoch != self.sampler.epoch:
            self.ds_train.set_epoch(self.sampler.epoch)
        self.epoch = self.sampler.epoch
        self.batch_cursor = self.sampler.batch_cursor

    def _next_batch(self):
        self._sync_epoch_to_dataset()
        if self.loader_iter is None:
            self.loader_iter = iter(self.loader)
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            # Iterator exhausted — ensure epoch/cursor are consistent, then rebuild.
            if self.sampler.num_batches() == 0:
                raise RuntimeError('ResumableBatchSampler has zero batches')
            if self.sampler.batch_cursor >= self.sampler.num_batches():
                self.sampler.epoch += 1
                self.sampler.batch_cursor = 0
                self.sampler.permutation = self.sampler._make_permutation(self.sampler.epoch)
            self.ds_train.set_epoch(self.sampler.epoch)
            self.loader = self._make_loader()
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)
        # Advance consumed cursor AFTER successful fetch
        rolled = self.sampler.advance(1)
        if rolled:
            self.ds_train.set_epoch(self.sampler.epoch)
            # Rebuild loader so the next iterator starts at new epoch cursor 0
            self.loader = self._make_loader()
            self.loader_iter = None
        self.epoch = self.sampler.epoch
        self.batch_cursor = self.sampler.batch_cursor
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
        aug = batch.get('aug_meta')
        aux = dict(
            sample_ids=list(batch.get('sample_id')),
            aug_meta=aug,
            pred_min=float(pred.detach().min().item()),
            pred_max=float(pred.detach().max().item()),
            pred_mean=float(pred.detach().mean().item()),
        )
        return loss, aux

    def _grads_finite(self) -> bool:
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
        return True

    def _grad_norms(self) -> dict:
        total = 0.0
        ref_branch = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = float(p.grad.data.float().norm().item())
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

    def train_update(self) -> dict:
        """One optimizer update = grad_accum micro-batches."""
        self.model.train()
        # LR for the upcoming update index (== current global_step)
        lrs = self.sched.apply_lr(self.global_step)
        self.opt.zero_grad(set_to_none=True)

        losses = []
        sample_ids = []
        aug_metas = []
        last_aux = {}
        for _ in range(self.accum):
            batch = self._next_batch()
            with autocast(enabled=self.amp):
                loss, aux = self._forward_loss(batch)
                loss_scaled = loss / float(self.accum)
            self.scaler.scale(loss_scaled).backward()
            self.micro_step += 1
            losses.append(float(loss.item()))
            sample_ids.extend(list(aux.get('sample_ids') or []))
            if aux.get('aug_meta') is not None:
                aug_metas.append(aux['aug_meta'])
            last_aux = aux

        self.scaler.unscale_(self.opt)
        finite = self._grads_finite()
        grad_info = {}
        did_step = False
        if finite:
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            grad_info = self._grad_norms()
            self.scaler.step(self.opt)
            did_step = True
            self.global_step += 1
            self.optimizer_updates = self.global_step
            self.step = self.global_step  # legacy alias
            self.sched.mark_update_done()
            # Prepare LR for the *next* update (optional; also done at loop start)
            if self.global_step < self.max_steps:
                self.sched.apply_lr(self.global_step)
        else:
            self.skipped_update_count += 1
            # clear grads; do not advance global_step / scheduler
            self.opt.zero_grad(set_to_none=True)
        self.scaler.update()

        peak = 0.0
        if str(self.device).startswith('cuda'):
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        log = dict(
            global_step=self.global_step,
            step=self.global_step,  # legacy == global_step
            micro_step=self.micro_step,
            optimizer_updates=self.optimizer_updates,
            skipped_update_count=self.skipped_update_count,
            epoch=self.epoch,
            batch_cursor=self.batch_cursor,
            grad_accum=self.accum,
            effective_batch_size=self.batch_size * self.accum,
            loss=float(sum(losses) / max(1, len(losses))),
            micro_losses=losses,
            learning_rate=float(lrs[0]),
            grad_scaler=float(self.scaler.get_scale()) if self.amp else 1.0,
            peak_memory=peak,
            prior_source=self.prior_source,
            did_optimizer_step=did_step,
            sample_ids=sample_ids,
            aug_meta=aug_metas,
            **self._alpha_beta(),
            **grad_info,
            pred_min=last_aux.get('pred_min'),
            pred_max=last_aux.get('pred_max'),
            pred_mean=last_aux.get('pred_mean'),
        )
        return log

    # Back-compat name used by older smoke scripts
    def train_micro_step(self) -> dict:
        return self.train_update()

    @torch.no_grad()
    def validate(self, max_images: Optional[int] = None) -> dict:
        self.model.eval()
        n = 0
        psnrs, ssims = [], []
        limit = max_images or int(self.tr.get('val_max_images', len(self.ds_val)))
        meta = {}
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
            global_step=self.global_step,
        )

    def state_dict_ckpt(self) -> dict:
        return dict(
            protocol_version=PROTOCOL_VERSION,
            model=self.model.state_dict(),
            optimizer=self.opt.state_dict(),
            scheduler=self.sched.state_dict(),
            scaler=self.scaler.state_dict(),
            global_step=self.global_step,
            step=self.global_step,
            micro_step=self.micro_step,
            optimizer_updates=self.optimizer_updates,
            skipped_update_count=self.skipped_update_count,
            epoch=self.epoch,
            batch_cursor=self.batch_cursor,
            sampler_state=self.sampler.state_dict(),
            best=self.best,
            best_psnr=self.best_psnr,
            cfg=self.cfg,
            train_protocol=self.train_protocol,
            model_family='asrt_irp',
            model_version='irp_v1',
            prior_source=self.prior_source,
            irp_hparams=extract_irp_hparams(self.cfg),
            train_manifest_hash=self.train_hash,
            val_manifest_hash=self.val_hash,
            training_config_hash=_stable_hash(self.train_protocol),
            model_config_hash=_stable_hash(extract_irp_hparams(self.cfg)),
            manifest_hash=self.train_hash,
            seed=self.seed,
            rng_state=capture_rng_state(),
            rng=capture_rng_state(),
            halo=self.halo,
            wrong_mapping_path=self.wrong_mapping_path,
            git_commit=self.git.get('git_commit'),
            git_dirty=self.git.get('git_dirty'),
        )

    def save_checkpoint(self, tag: str):
        """Only valid on update boundaries (caller responsibility)."""
        path = self.out_dir / 'checkpoints' / ('%s.pt' % tag)
        torch.save(self.state_dict_ckpt(), path)
        return str(path)

    def load_checkpoint(self, path: str, *, weights_only: bool = False):
        ck = torch.load(path, map_location='cpu')
        if weights_only:
            self.model.load_state_dict(ck['model'], strict=True)
            return ck

        # Model family / hparams
        ck_hp = ck.get('irp_hparams') or extract_irp_hparams(ck.get('cfg') or {})
        model_hp = extract_irp_hparams(self.cfg)
        conflicts = compare_irp_hparams(ck_hp, model_hp)
        if conflicts:
            raise RuntimeError(
                'IRP hparam mismatch on resume (not weights-only): %s' % conflicts)

        # Training protocol
        ck_proto = ck.get('train_protocol')
        if ck_proto is None:
            raise RuntimeError(
                'checkpoint missing train_protocol; refuse silent resume '
                '(use weights_only=True for fine-tune load)')
        proto_conflicts = compare_train_protocol(ck_proto, self.train_protocol)
        if proto_conflicts:
            raise RuntimeError(
                'train protocol mismatch on resume: %s' % proto_conflicts)

        if str(ck.get('protocol_version', '')) and str(
                ck.get('protocol_version')) != PROTOCOL_VERSION:
            # Allow loading v2-only; warn via exception if completely missing semantics
            if 'global_step' not in ck and 'sampler_state' not in ck:
                raise RuntimeError(
                    'unsupported checkpoint protocol_version=%r'
                    % ck.get('protocol_version'))

        self.model.load_state_dict(ck['model'], strict=True)
        self.opt.load_state_dict(ck['optimizer'])
        self.sched.load_state_dict(ck['scheduler'])
        if 'scaler' in ck:
            self.scaler.load_state_dict(ck['scaler'])

        self.global_step = int(ck.get('global_step', ck.get('step', ck.get('optimizer_updates', 0))))
        self.step = self.global_step
        self.optimizer_updates = int(ck.get('optimizer_updates', self.global_step))
        self.micro_step = int(ck.get('micro_step', 0))
        self.skipped_update_count = int(ck.get('skipped_update_count', 0))
        self.best = ck.get('best')
        self.best_psnr = ck.get('best_psnr')

        if 'sampler_state' in ck:
            self.sampler.load_state_dict(ck['sampler_state'])
        else:
            raise RuntimeError('checkpoint missing sampler_state; cannot resume data position')
        self.ds_train.set_epoch(self.sampler.epoch)
        self.epoch = self.sampler.epoch
        self.batch_cursor = self.sampler.batch_cursor
        restore_rng_state(ck.get('rng_state') or ck.get('rng'))
        self.loader = self._make_loader()
        self.loader_iter = None
        # Ensure LR matches next update
        self.sched.apply_lr(self.global_step)
        return ck

    def fit(self):
        val_every = int(self.tr.get('val_every', 1000))
        ckpt_every = int(self.tr.get('ckpt_every', 2000))
        log_every = int(self.tr.get('log_every', 50))
        while self.global_step < self.max_steps:
            log = self.train_update()
            if (log['did_optimizer_step'] and (
                    self.global_step % log_every == 0 or self.global_step == 1)):
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(log, default=str) + '\n')
                print('global_step', self.global_step, 'loss', log['loss'],
                      'lr', log['learning_rate'], flush=True)
            if (log['did_optimizer_step'] and val_every > 0
                    and self.global_step % val_every == 0):
                metrics = self.validate()
                with open(self.out_dir / 'val_log.jsonl', 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                if self.best_psnr is None or metrics['psnr'] > self.best_psnr:
                    self.best_psnr = metrics['psnr']
                    self.best = metrics
                    self.save_checkpoint('best_validation_psnr')
            if (log['did_optimizer_step'] and ckpt_every > 0
                    and self.global_step % ckpt_every == 0):
                self.save_checkpoint('periodic_%d' % self.global_step)
            # If AMP skipped, continue without counting toward max_steps progress
            # (global_step unchanged). Avoid infinite loop if permanently NaN:
            if (not log['did_optimizer_step']
                    and self.skipped_update_count > 100
                    and self.skipped_update_count % 100 == 0):
                print('WARNING: repeated AMP skipped updates:',
                      self.skipped_update_count, flush=True)
        self.save_checkpoint('last')
        return dict(
            best=self.best, best_psnr=self.best_psnr,
            global_step=self.global_step, step=self.global_step)
