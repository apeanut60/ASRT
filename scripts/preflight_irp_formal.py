#!/usr/bin/env python
"""GPU preflight for formal A2/A3 (does NOT initialize formal training)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asrt.models.asrt_irp import build_asrt_irp  # noqa: E402
from asrt.training.irp_trainer import IRPTrainer, git_meta  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402


def _load(path):
    return yaml.safe_load(open(path))


def check_init_equality(cfg_a2, cfg_a3):
    seed_everything(42)
    a2 = build_asrt_irp(cfg_a2)
    seed_everything(42)
    a3 = build_asrt_irp(cfg_a3)
    sd2, sd3 = a2.state_dict(), a3.state_dict()
    assert set(sd2) == set(sd3)
    for k in sd2:
        assert torch.equal(sd2[k], sd3[k]), k
    n2 = sum(p.numel() for p in a2.parameters())
    n3 = sum(p.numel() for p in a3.parameters())
    assert n2 == n3
    return dict(
        params=n2,
        keys=len(sd2),
        prior_a2=str(a2.prior_source),
        prior_a3=str(a3.prior_source),
        equal=True,
    )


def run_one(cfg, tag, steps, out_root):
    cfg = copy.deepcopy(cfg)
    cfg['out_dir'] = str(out_root / tag)
    cfg['training']['max_steps'] = int(steps)
    cfg['training']['val_every'] = max(1, int(steps))
    cfg['training']['ckpt_every'] = max(1, int(steps))
    cfg['training']['log_every'] = 1
    cfg['training']['val_max_images'] = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['training']['device'] = device
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    seed_everything(42)
    tr = IRPTrainer(cfg, device=device)
    logs = []
    for _ in range(steps):
        log = tr.train_update()
        logs.append(log)
        if not torch.isfinite(torch.tensor(log['loss'])):
            raise RuntimeError('non-finite loss')
        if log.get('gradient_norm') is not None and not (
                log['gradient_norm'] == log['gradient_norm']):
            raise RuntimeError('non-finite grad')
    ck = tr.save_checkpoint('preflight_last')
    # Checkpoint roundtrip into a sibling empty out_dir
    cfg_r = copy.deepcopy(cfg)
    cfg_r['out_dir'] = str(out_root / (tag + '_resume'))
    seed_everything(0)
    tr_r = IRPTrainer(cfg_r, device=device)
    # Allow loading into this fresh dir's trainer state from ck of sibling run
    tr_r.load_checkpoint(ck)
    assert tr_r.global_step == steps
    val = tr_r.validate(max_images=2)
    peak = 0.0
    if device == 'cuda':
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    ref_g = [l.get('reference_branch_grad_norm', 0.0) for l in logs]
    return dict(
        tag=tag,
        prior=tr.prior_source,
        steps=tr.global_step,
        halo=tr.halo,
        batch_size=tr.batch_size,
        grad_accum=tr.accum,
        effective_batch=tr.batch_size * tr.accum,
        amp=tr.amp,
        device=device,
        losses=[l['loss'] for l in logs],
        lrs=[l['learning_rate'] for l in logs],
        grad_norms=[l.get('gradient_norm') for l in logs],
        ref_grad_norms=ref_g,
        ref_grad_nonzero=any((g or 0) > 0 for g in ref_g),
        alphas=[{k: l[k] for k in l if k.startswith('alpha_') or k.startswith('beta_')}
                for l in logs[:3]],
        val=val,
        peak_memory_mb=peak,
        checkpoint=ck,
        any_nan=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a2', default=str(ROOT / 'configs/asrt_irp_v1_a2.yaml'))
    ap.add_argument('--a3', default=str(ROOT / 'configs/asrt_irp_v1_a3.yaml'))
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--out', default='/root/data/experiments/ASRT/vnext_v1/preflight')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', git=git_meta(), errors=[])
    try:
        cfg_a2 = _load(a.a2)
        cfg_a3 = _load(a.a3)
        # configs must differ only by prior_source + out_dir
        c2, c3 = copy.deepcopy(cfg_a2), copy.deepcopy(cfg_a3)
        assert c2['model']['prior_source'] == 'low'
        assert c3['model']['prior_source'] == 'nano'
        c2['model'].pop('prior_source'); c3['model'].pop('prior_source')
        c2.pop('out_dir'); c3.pop('out_dir')
        assert json.dumps(c2, sort_keys=True) == json.dumps(c3, sort_keys=True), (
            'A2/A3 configs differ beyond prior_source/out_dir')
        report['config_parity'] = True
        report['init'] = check_init_equality(cfg_a2, cfg_a3)
        print('init', report['init'], flush=True)

        # Try batch_size=8 first; on OOM fall back keeping effective batch=8
        batch_plan = [(8, 1), (4, 2), (2, 4)]
        chosen = None
        last_err = None
        for bs, accum in batch_plan:
            try:
                print('trying batch=%d accum=%d' % (bs, accum), flush=True)
                cfg_try = copy.deepcopy(cfg_a2)
                cfg_try['training']['batch_size'] = bs
                cfg_try['training']['grad_accum'] = accum
                # disposable probe
                probe_out = out / ('probe_bs%d_a%d' % (bs, accum))
                if probe_out.exists():
                    import shutil
                    shutil.rmtree(probe_out)
                r = run_one(cfg_try, 'probe_bs%d_a%d' % (bs, accum), min(2, a.steps), out)
                chosen = (bs, accum, r['peak_memory_mb'])
                print('probe ok', chosen, flush=True)
                break
            except RuntimeError as e:
                last_err = e
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    print('OOM at batch', bs, flush=True)
                    continue
                raise
        if chosen is None:
            raise RuntimeError('no batch plan worked: %s' % last_err)
        bs, accum, _ = chosen
        report['batch_plan'] = dict(batch_size=bs, grad_accum=accum,
                                    effective_batch=bs * accum,
                                    peak_probe_mb=chosen[2])

        # Apply chosen plan to both
        for cfg in (cfg_a2, cfg_a3):
            cfg['training']['batch_size'] = bs
            cfg['training']['grad_accum'] = accum

        report['a2'] = run_one(cfg_a2, 'a2_%du' % a.steps, a.steps, out)
        print('A2 preflight', report['a2']['peak_memory_mb'], flush=True)
        report['a3'] = run_one(cfg_a3, 'a3_%du' % a.steps, a.steps, out)
        print('A3 preflight', report['a3']['peak_memory_mb'], flush=True)

        # Fingerprint parity on train/val hashes from preflight dirs
        fp2 = json.load(open(out / ('a2_%du' % a.steps) / 'data_fingerprint.json'))
        fp3 = json.load(open(out / ('a3_%du' % a.steps) / 'data_fingerprint.json'))
        report['fingerprint_match'] = (
            fp2['train_manifest_sha256'] == fp3['train_manifest_sha256']
            and fp2['val_manifest_sha256'] == fp3['val_manifest_sha256']
            and fp2['sample_count'] == fp3['sample_count']
        )
        assert report['fingerprint_match']
        report['status'] = 'PASS'
        # Write recommended batch into a sidecar for formal launch
        with open(out / 'recommended_batch.json', 'w') as f:
            json.dump(report['batch_plan'], f, indent=2)
    except Exception as e:
        report['status'] = 'FAIL'
        report['errors'].append(str(e))
        traceback.print_exc()
    path = out / 'preflight_report.json'
    path.write_text(json.dumps(report, indent=2, default=str))
    print('WROTE', path, 'STATUS', report['status'], flush=True)
    sys.exit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
