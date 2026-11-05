#!/usr/bin/env python
"""Phase I–IV diagnostics for IRP gate collapse (no training)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asrt.data.irp_dataset import TripletIRPDataset  # noqa: E402
from asrt.data.reference_modes import build_wrong_mapping  # noqa: E402
from asrt.evaluation.metrics import psnr_ssim_rgb  # noqa: E402
from asrt.evaluation.protocol import infer_with_protocol, resolve_eval_cfg  # noqa: E402
from asrt.models.asrt_irp import build_asrt_irp, extract_irp_hparams  # noqa: E402
from asrt.models.factory import load_checkpoint_strict  # noqa: E402
from asrt.training.irp_trainer import _load_rows, split_irp_gate_params  # noqa: E402

PIXEL_CHANGE_THRESH = 1e-5


def audit_experiment(exp_dir: Path) -> dict:
    exp_dir = Path(exp_dir)
    out = dict(path=str(exp_dir), exists=exp_dir.is_dir())
    for name in ('train_log.jsonl', 'val_log.jsonl', 'resolved_config.json',
                 'checkpoints/best_validation_psnr.pt', 'checkpoints/last.pt'):
        p = exp_dir / name
        out[name] = dict(exists=p.exists(), size=p.stat().st_size if p.exists() else 0)
    if not (exp_dir / 'resolved_config.json').exists():
        return out
    rc = json.load(open(exp_dir / 'resolved_config.json'))
    cfg = rc.get('cfg') or {}
    out.update(dict(
        git=rc.get('git'),
        protocol_version=rc.get('protocol_version'),
        prior_source=(cfg.get('model') or {}).get('prior_source'),
        seed=(cfg.get('training') or {}).get('seed'),
        train_manifest_hash=rc.get('train_manifest_hash'),
        val_manifest_hash=rc.get('val_manifest_hash'),
        halo=rc.get('halo'),
        config_hash=rc.get('config_hash'),
    ))
    for tag in ('best_validation_psnr', 'last'):
        ck_path = exp_dir / 'checkpoints' / ('%s.pt' % tag)
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, map_location='cpu')
        out['ckpt_%s' % tag] = dict(
            global_step=ck.get('global_step', ck.get('step')),
            prior_source=ck.get('prior_source'),
            model_family=ck.get('model_family'),
            model_version=ck.get('model_version'),
            irp_hparams=ck.get('irp_hparams'),
            train_manifest_hash=ck.get('train_manifest_hash'),
            n_params=sum(v.numel() for v in ck['model'].values()
                         if torch.is_tensor(v)),
            has_optimizer='optimizer' in ck,
            gate_values=_gate_values_from_state(ck['model']),
        )
    # gate timeline from train log
    tl = exp_dir / 'train_log.jsonl'
    if tl.exists():
        rows = [json.loads(l) for l in open(tl) if l.strip()]
        out['train_log_n'] = len(rows)
        out['train_last_step'] = rows[-1].get('global_step') if rows else None
        out['gate_timeline'] = _gate_timeline(rows)
    return out


def _gate_values_from_state(state: dict) -> dict:
    keys = {
        'alpha_h4': 'backbone.coarse_irp.alpha',
        'alpha_h2': 'backbone.middle_irp.alpha',
        'alpha_h1': 'backbone.fine_irp.alpha',
        'beta_h2': 'backbone.middle_irp.beta',
        'beta_h1': 'backbone.fine_irp.beta',
    }
    out = {}
    for tag, k in keys.items():
        if k in state:
            out[tag] = float(state[k].reshape(-1)[0].item())
    return out


def _gate_timeline(rows):
    marks = {}
    for thr_name, thr in (('1e-3', 1e-3), ('1e-6', 1e-6)):
        for key in ('alpha_h4', 'alpha_h2', 'alpha_h1', 'beta_h2', 'beta_h1'):
            for r in rows:
                v = r.get(key)
                if v is None:
                    continue
                if abs(float(v)) < thr:
                    marks['%s_lt_%s' % (key, thr_name)] = r.get('global_step')
                    break
    return marks


def load_model_identity(cfg_path, ckpt_path, device, expect_prior):
    cfg = yaml.safe_load(open(cfg_path))
    assert str(cfg['model']['prior_source']).lower() == str(expect_prior).lower()
    model = build_asrt_irp(cfg).to(device)
    report = load_checkpoint_strict(
        model, ckpt_path, device='cpu', expect_version='irp_v1',
        external_cfg=cfg, allow_hparam_override=False)
    model.eval()
    ck = torch.load(ckpt_path, map_location='cpu')
    return model, cfg, ck, report


@torch.no_grad()
def run_interventions(model, cfg, rows, device, wrong_map, ckpt_step, out_rows):
    modes = ['correct', 'wrong', 'patch_shuffle', 'pixel_shuffle', 'ref_low', 'noref']
    id_to_row = {str(r['sample_id']): r for r in rows}
    preds_correct = {}
    metrics = {m: [] for m in modes}

    for mode in modes:
        ds = TripletIRPDataset(
            rows, crop=0, seed=0, augment=False, halo=0,
            require_ref=True, ref_mode=mode, wrong_mapping=wrong_map,
            id_to_row=id_to_row,
        )
        ev = resolve_eval_cfg(cfg)
        for i in range(len(ds)):
            b = ds[i]
            sid = str(b['sample_id'])
            low = b['low'][None].to(device)
            gt = b['gt'][None].to(device)
            ref = b['ref']
            if ref is not None:
                ref = ref[None].to(device)
            # NoRef: pass ref=None while keeping prior_source=nano
            if mode == 'noref':
                pred, meta = infer_with_protocol(model, low, None, {'evaluation': cfg.get('evaluation')})
            else:
                pred, meta = infer_with_protocol(model, low, ref, {'evaluation': cfg.get('evaluation')})
            m = psnr_ssim_rgb(pred, gt)
            rec = dict(
                sample_id=sid,
                checkpoint_step=int(ckpt_step),
                reference_mode=mode,
                psnr=float(m['psnr_rgb']),
                ssim=float(m['ssim_rgb']),
                inference_used=meta.get('inference_used'),
            )
            if mode == 'correct':
                preds_correct[sid] = pred.detach().cpu()
            else:
                pc = preds_correct[sid]
                diff = (pred.detach().cpu() - pc).float()
                absdiff = diff.abs()
                rec.update(dict(
                    output_mae_vs_correct=float(absdiff.mean().item()),
                    output_rmse_vs_correct=float(torch.sqrt((diff * diff).mean()).item()),
                    output_max_abs_vs_correct=float(absdiff.max().item()),
                    output_changed_pixel_ratio=float(
                        (absdiff > PIXEL_CHANGE_THRESH).float().mean().item()),
                    pixel_change_threshold=PIXEL_CHANGE_THRESH,
                ))
            metrics[mode].append(rec)
            out_rows.append(rec)

    # fill deltas vs correct for all modes
    by_c = {r['sample_id']: r for r in metrics['correct']}
    for mode in modes:
        for r in metrics[mode]:
            c = by_c[r['sample_id']]
            r['delta_psnr_vs_correct'] = float(r['psnr'] - c['psnr'])
            r['delta_ssim_vs_correct'] = float(r['ssim'] - c['ssim'])
            if mode == 'correct':
                r.update(dict(
                    output_mae_vs_correct=0.0,
                    output_rmse_vs_correct=0.0,
                    output_max_abs_vs_correct=0.0,
                    output_changed_pixel_ratio=0.0,
                    pixel_change_threshold=PIXEL_CHANGE_THRESH,
                ))
    summary = {}
    for mode in modes:
        rows_m = metrics[mode]
        summary[mode] = dict(
            mean_psnr=float(np.mean([r['psnr'] for r in rows_m])),
            mean_ssim=float(np.mean([r['ssim'] for r in rows_m])),
            mean_delta_psnr=float(np.mean([r['delta_psnr_vs_correct'] for r in rows_m])),
            mean_mae_vs_correct=float(np.mean([r['output_mae_vs_correct'] for r in rows_m])),
            mean_rmse_vs_correct=float(np.mean([r['output_rmse_vs_correct'] for r in rows_m])),
            mean_changed_pixel_ratio=float(
                np.mean([r['output_changed_pixel_ratio'] for r in rows_m])),
            max_mae_vs_correct=float(np.max([r['output_mae_vs_correct'] for r in rows_m])),
        )
    return summary


@torch.no_grad()
def branch_contributions(model, cfg, rows, device, n_images=8):
    """Measure branch stats + Ref ON vs OFF output diffs."""
    ds = TripletIRPDataset(
        rows[:n_images], crop=0, seed=0, augment=False, halo=0,
        require_ref=True, ref_mode='correct',
        id_to_row={str(r['sample_id']): r for r in rows},
    )
    records = []
    for i in range(len(ds)):
        b = ds[i]
        low = b['low'][None].to(device)
        ref = b['ref'][None].to(device)
        pred_on, aux_on = model(low, ref, return_aux=True, disable_reference=False)
        pred_off, aux_off = model(low, ref, return_aux=True, disable_reference=True)
        diff = (pred_on - pred_off).float()
        absdiff = diff.abs()
        rec = dict(
            sample_id=str(b['sample_id']),
            mae_on_vs_off=float(absdiff.mean().item()),
            rmse_on_vs_off=float(torch.sqrt((diff * diff).mean()).item()),
            max_abs_on_vs_off=float(absdiff.max().item()),
            changed_pixel_ratio=float((absdiff > PIXEL_CHANGE_THRESH).float().mean().item()),
            pixel_change_threshold=PIXEL_CHANGE_THRESH,
        )
        for tag, st in (aux_on.get('branch_stats') or {}).items():
            for k, v in st.items():
                rec['%s_%s_on' % (tag, k)] = v
        for tag, st in (aux_off.get('branch_stats') or {}).items():
            rec['%s_ref_contribution_ratio_off' % tag] = st.get('ref_contribution_ratio')
            rec['%s_beta_ref_mod_l2_off' % tag] = st.get('beta_ref_mod_l2')
        records.append(rec)
    # weight integrity: compare state dict hash before/after already same model
    return records


def audit_optimizer_gates(ckpt_path: Path) -> dict:
    ck = torch.load(ckpt_path, map_location='cpu')
    model = build_asrt_irp(ck['cfg'])
    model.load_state_dict(ck['model'], strict=True)
    gate_params, other_params, gate_names = split_irp_gate_params(model)
    opt = ck.get('optimizer') or {}
    state = opt.get('state') or {}
    # Map parameter id → name via rebuilding optimizer? state is keyed by param id
    # which is not stable across loads. Instead match by comparing tensor data from
    # model params that are gates and look up via param group param indices if saved.
    # PyTorch Adam state_dict uses integer indices into param_groups flattened list.
    report = dict(
        gate_names=gate_names,
        n_gate=len(gate_params),
        n_other=len(other_params),
        training_wd=float((ck.get('cfg') or {}).get('training', {}).get('weight_decay', 1e-4)),
        training_lr=float((ck.get('cfg') or {}).get('training', {}).get('learning_rate', 1e-4)),
        opt_n_param_groups=len(opt.get('param_groups') or []),
        gate_adam_state=[],
        note='V1 used a single Adam group: gates share lr/weight_decay with all params.',
    )
    # Reconstruct param order as in state_dict save: same as model.parameters() order
    # of requires_grad params when optimizer was built (all trainable).
    trainable = [p for p in model.parameters() if p.requires_grad]
    # After load_state_dict, optimizer state keys are old ids — we only have
    # state_dict with integer keys. Map via param_groups[].params lists of indices.
    # When torch.save(opt.state_dict()), state keys are param ids; load gives
    # remapped? Actually saved state uses raw id() which is NOT portable.
    # Check what's in the checkpoint.
    state_keys = list(state.keys())
    report['optimizer_state_key_types'] = [str(type(k).__name__) for k in state_keys[:3]]
    report['optimizer_state_n'] = len(state)
    # Prefer param_groups metadata
    pgs = opt.get('param_groups') or []
    if len(pgs) == 1:
        report['single_group'] = dict(
            lr=pgs[0].get('lr'),
            weight_decay=pgs[0].get('weight_decay'),
            betas=pgs[0].get('betas'),
            n_params=len(pgs[0].get('params') or []),
        )
        report['gates_share_weight_decay'] = True
        report['gate_weight_decay'] = pgs[0].get('weight_decay')
    # Extract gate values + whether Adam moments exist by matching shapes in state
    # Fall back: report gate tensor values from model weights (already known collapsed)
    report['gate_values'] = _gate_values_from_state(ck['model'])
    # Try to find adam state entries whose exp_avg has shape [] (scalars)
    scalar_states = []
    for k, v in state.items():
        ea = v.get('exp_avg') if isinstance(v, dict) else None
        if torch.is_tensor(ea) and ea.numel() == 1:
            scalar_states.append(dict(
                key=str(k),
                step=int(v.get('step', -1)) if not torch.is_tensor(v.get('step'))
                else int(v['step'].item()),
                exp_avg=float(ea.reshape(-1)[0].item()),
                exp_avg_sq=float(v['exp_avg_sq'].reshape(-1)[0].item()),
            ))
    report['scalar_adam_states'] = scalar_states
    report['n_scalar_adam_states'] = len(scalar_states)
    return report


def image_paired_bootstrap(a2_ckpt, a3_ckpt, a2_cfg, a3_cfg, rows, device):
    """Image-level paired bootstrap on best ckpts (correct protocol)."""
    m2, cfg2, ck2, _ = load_model_identity(a2_cfg, a2_ckpt, device, 'low')
    m3, cfg3, ck3, _ = load_model_identity(a3_cfg, a3_ckpt, device, 'nano')
    ds = TripletIRPDataset(
        rows, crop=0, seed=0, augment=False, halo=0, require_ref=True,
        ref_mode='correct', id_to_row={str(r['sample_id']): r for r in rows})
    paired = []
    for i in range(len(ds)):
        b = ds[i]
        low = b['low'][None].to(device)
        gt = b['gt'][None].to(device)
        ref = b['ref'][None].to(device)
        p2, _ = infer_with_protocol(m2, low, ref, {'evaluation': cfg2.get('evaluation')})
        p3, _ = infer_with_protocol(m3, low, ref, {'evaluation': cfg3.get('evaluation')})
        m2m = psnr_ssim_rgb(p2, gt)
        m3m = psnr_ssim_rgb(p3, gt)
        paired.append(dict(
            sample_id=str(b['sample_id']),
            A2_PSNR=float(m2m['psnr_rgb']), A3_PSNR=float(m3m['psnr_rgb']),
            Delta_PSNR=float(m3m['psnr_rgb'] - m2m['psnr_rgb']),
            A2_SSIM=float(m2m['ssim_rgb']), A3_SSIM=float(m3m['ssim_rgb']),
            Delta_SSIM=float(m3m['ssim_rgb'] - m2m['ssim_rgb']),
        ))
    d = np.array([r['Delta_PSNR'] for r in paired])
    rng = np.random.RandomState(0)
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(5000)]
    summary = dict(
        n_images=len(paired),
        mean_delta_psnr=float(d.mean()),
        median_delta_psnr=float(np.median(d)),
        bootstrap_ci95=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        positive_ratio=float((d > 0).mean()),
        negative_ratio=float((d < 0).mean()),
        a2_step=ck2.get('global_step'), a3_step=ck3.get('global_step'),
        note='Bootstrap over validation IMAGES for best checkpoints; '
             'NOT over training-time val checkpoints.',
    )
    return summary, paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/data/experiments/ASRT/vnext_v1/irp_gate_diagnostics')
    ap.add_argument('--a2_dir', default='/root/data/experiments/ASRT/vnext_v1/A2_low_prior_seed42')
    ap.add_argument('--a3_dir', default='/root/data/experiments/ASRT/vnext_v1/A3_nano_prior_seed42')
    ap.add_argument('--a2_cfg', default=str(ROOT / 'configs/asrt_irp_v1_a2.yaml'))
    ap.add_argument('--a3_cfg', default=str(ROOT / 'configs/asrt_irp_v1_a3.yaml'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'plots').mkdir(parents=True, exist_ok=True)

    audit = dict(
        a2=audit_experiment(a.a2_dir),
        a3=audit_experiment(a.a3_dir),
    )
    with open(out / 'checkpoint_audit.json', 'w') as f:
        json.dump(audit, f, indent=2, default=str)
    print('Wrote checkpoint_audit.json', flush=True)

    # Negative identity check
    neg = {}
    try:
        load_model_identity(a.a2_cfg, str(Path(a.a3_dir) / 'checkpoints/best_validation_psnr.pt'),
                            a.device, 'low')
        neg['a3_into_a2'] = 'UNEXPECTED_PASS'
    except Exception as e:
        neg['a3_into_a2'] = 'REJECTED:%s' % type(e).__name__
    try:
        load_model_identity(a.a3_cfg, str(Path(a.a2_dir) / 'checkpoints/best_validation_psnr.pt'),
                            a.device, 'nano')
        neg['a2_into_a3'] = 'UNEXPECTED_PASS'
    except Exception as e:
        neg['a2_into_a3'] = 'REJECTED:%s' % type(e).__name__
    audit['negative_load_tests'] = neg
    with open(out / 'checkpoint_audit.json', 'w') as f:
        json.dump(audit, f, indent=2, default=str)
    print('negative tests', neg, flush=True)

    cfg3 = yaml.safe_load(open(a.a3_cfg))
    rows = _load_rows(cfg3, 'val')
    wrong_map = build_wrong_mapping(
        [str(r['sample_id']) for r in rows], seed=42,
        out_path=str(out / 'wrong_ref_mapping_val.json'))

    all_rows = []
    summaries = {}
    for tag in ('best_validation_psnr', 'last'):
        ck = Path(a.a3_dir) / 'checkpoints' / ('%s.pt' % tag)
        model, cfg, ckblob, rep = load_model_identity(
            a.a3_cfg, str(ck), a.device, 'nano')
        assert str(model.prior_source).lower() == 'nano'
        step = int(ckblob.get('global_step', ckblob.get('step', -1)))
        print('interventions', tag, 'step', step, flush=True)
        summaries[tag] = run_interventions(
            model, cfg, rows, a.device, wrong_map, step, all_rows)
        # branch contributions
        br = branch_contributions(model, cfg, rows, a.device, n_images=min(8, len(rows)))
        with open(out / ('branch_contributions_%s.json' % tag), 'w') as f:
            json.dump(br, f, indent=2)
        # mean ref contribution
        summaries[tag]['branch_mean_rho_R_h2'] = float(np.mean(
            [r.get('h2_ref_contribution_ratio_on', 0) for r in br]))
        summaries[tag]['branch_mean_rho_R_h1'] = float(np.mean(
            [r.get('h1_ref_contribution_ratio_on', 0) for r in br]))
        summaries[tag]['branch_mean_mae_on_vs_off'] = float(np.mean(
            [r['mae_on_vs_off'] for r in br]))
        del model
        torch.cuda.empty_cache()

    with open(out / 'reference_interventions.json', 'w') as f:
        json.dump(dict(summary=summaries, pixel_change_threshold=PIXEL_CHANGE_THRESH,
                       per_image=all_rows), f, indent=2)
    with open(out / 'reference_interventions.csv', 'w', newline='') as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    # flatten branch csv
    br_all = []
    for tag in ('best_validation_psnr', 'last'):
        br = json.load(open(out / ('branch_contributions_%s.json' % tag)))
        for r in br:
            r = dict(r)
            r['checkpoint'] = tag
            br_all.append(r)
    with open(out / 'branch_contributions.csv', 'w', newline='') as f:
        if br_all:
            w = csv.DictWriter(f, fieldnames=sorted(br_all[0].keys()))
            w.writeheader()
            w.writerows(br_all)
    print('interventions summary', json.dumps(summaries, indent=2), flush=True)

    # optimizer audit on A3 last (has optimizer state)
    opt_audit = dict(
        a3_last=audit_optimizer_gates(Path(a.a3_dir) / 'checkpoints/last.pt'),
        a3_best=audit_optimizer_gates(
            Path(a.a3_dir) / 'checkpoints/best_validation_psnr.pt'),
        a2_last=audit_optimizer_gates(Path(a.a2_dir) / 'checkpoints/last.pt'),
    )
    with open(out / 'gate_optimizer_audit.json', 'w') as f:
        json.dump(opt_audit, f, indent=2, default=str)
    print('optimizer audit gates_share_wd',
          opt_audit['a3_last'].get('gates_share_weight_decay'), flush=True)

    # image-level paired bootstrap best vs best
    print('image-level paired bootstrap...', flush=True)
    boot_sum, boot_paired = image_paired_bootstrap(
        str(Path(a.a2_dir) / 'checkpoints/best_validation_psnr.pt'),
        str(Path(a.a3_dir) / 'checkpoints/best_validation_psnr.pt'),
        a.a2_cfg, a.a3_cfg, rows, a.device)
    with open(out / 'best_ckpt_image_paired.json', 'w') as f:
        json.dump(dict(summary=boot_sum, per_image=boot_paired), f, indent=2)
    print('bootstrap', boot_sum, flush=True)
    print('DONE diagnostics ->', out)


if __name__ == '__main__':
    main()
