#!/usr/bin/env python
"""B0/B1/B2 10k-update gate ablations (nano prior). Does not touch V1 ckpts."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asrt.training.irp_trainer import IRPTrainer, split_irp_gate_params  # noqa: E402
from asrt.training.seed_utils import seed_everything  # noqa: E402


EXPS = [
    # B0: identical to V1 single-group Adam (gate_* unset → single group)
    dict(tag='B0_v1_equiv', gate_lr=None, gate_weight_decay=None),
    # B1: gates keep lr, zero weight decay
    dict(tag='B1_gate_wd0', gate_lr=1e-4, gate_weight_decay=0.0),
    # B2: lower gate lr, zero weight decay
    dict(tag='B2_gate_lr1e5_wd0', gate_lr=1e-5, gate_weight_decay=0.0),
]


def build_cfg(base_cfg, tag, gate_lr, gate_wd, out_root, max_steps):
    cfg = copy.deepcopy(base_cfg)
    cfg['out_dir'] = str(Path(out_root) / tag)
    cfg['model']['prior_source'] = 'nano'
    cfg['data']['ref_mode'] = 'correct'
    tr = cfg.setdefault('training', {})
    tr.update(dict(
        max_steps=int(max_steps),
        batch_size=8,
        grad_accum=1,
        crop_size=128,
        halo=None,
        amp=True,
        warmup_steps=1000,
        val_every=2000,
        ckpt_every=2000,
        log_every=50,
        num_workers=4,
        seed=42,
        learning_rate=1e-4,
        weight_decay=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        drop_last=True,
        val_max_images=32,
    ))
    if gate_lr is not None:
        tr['gate_lr'] = float(gate_lr)
    else:
        tr.pop('gate_lr', None)
    if gate_wd is not None:
        tr['gate_weight_decay'] = float(gate_wd)
    else:
        tr.pop('gate_weight_decay', None)
    return cfg


def summarize_run(out_dir: Path) -> dict:
    rows = [json.loads(l) for l in open(out_dir / 'train_log.jsonl') if l.strip()]
    val = []
    vp = out_dir / 'val_log.jsonl'
    if vp.exists():
        val = [json.loads(l) for l in open(vp) if l.strip()]
    def first_lt(key, thr):
        for r in rows:
            v = r.get(key)
            if v is not None and abs(float(v)) < thr:
                return r['global_step']
        return None
    last = rows[-1] if rows else {}
    return dict(
        out_dir=str(out_dir),
        final_step=last.get('global_step'),
        final_loss=last.get('loss'),
        opt_param_groups_mode=last.get('opt_param_groups_mode'),
        gate_lr=last.get('gate_lr'),
        gate_weight_decay=last.get('gate_weight_decay'),
        final_gates={k: last.get(k) for k in (
            'alpha_h4', 'alpha_h2', 'alpha_h1', 'beta_h2', 'beta_h1')},
        beta_lt_1e3_step=first_lt('beta_h2', 1e-3),
        beta_lt_1e6_step=first_lt('beta_h2', 1e-6),
        alpha_lt_1e3_step=first_lt('alpha_h4', 1e-3),
        alpha_lt_1e6_step=first_lt('alpha_h4', 1e-6),
        final_ref_grad=last.get('reference_branch_grad_norm'),
        final_gate_beta_grad=last.get('gate_beta_grad_norm'),
        val=val,
        best_val_psnr=(max(val, key=lambda x: x['psnr'])['psnr'] if val else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_config', default=str(ROOT / 'configs/asrt_irp_v1_a3.yaml'))
    ap.add_argument('--out_root',
                    default='/root/data/experiments/ASRT/vnext_v1/irp_gate_diagnostics/ablations')
    ap.add_argument('--max_steps', type=int, default=10000)
    ap.add_argument('--only', default=None, help='Run a single tag, e.g. B0_v1_equiv')
    a = ap.parse_args()
    base = yaml.safe_load(open(a.base_config))
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = dict(max_steps=a.max_steps, experiments={})

    for exp in EXPS:
        if a.only and exp['tag'] != a.only:
            continue
        tag = exp['tag']
        print('=== START', tag, '===', flush=True)
        cfg = build_cfg(base, tag, exp['gate_lr'], exp['gate_weight_decay'],
                        out_root, a.max_steps)
        seed_everything(42)
        tr = IRPTrainer(cfg, device=cfg['training']['device'])
        # Verify param group mode
        gate_p, other_p, names = split_irp_gate_params(tr.model)
        print(tag, 'opt_mode', tr.opt_param_groups_mode,
              'n_gates', len(names), 'n_groups', len(tr.opt.param_groups), flush=True)
        result = tr.fit()
        s = summarize_run(Path(cfg['out_dir']))
        s['fit_result'] = result
        s['gate_names'] = names
        summary['experiments'][tag] = s
        with open(out_root / 'short_ablation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print('=== DONE', tag, 'beta_lt_1e3', s['beta_lt_1e3_step'],
              'final_beta', s['final_gates'].get('beta_h2'), flush=True)
        del tr
        torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
