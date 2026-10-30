# ASRT-v4 Training Report

## Setup

| Item | Value |
|---|---|
| Config | `configs/stage_b_v4.yaml` |
| Out dir | `/root/data/experiments/ASRT/stage_b_v4/` |
| Init | Stage A `best.pt` @ 58k (weights only) |
| Manifests | `manifests_b_v2` (clean Real+Syn) |
| Dual forward | `λ_no_ref=0.5` |
| Budget | 100k steps |
| LR | backbone 1e-5 / v4_new 1e-4 / igab&illum 2e-5 |
| AMP | FP16 GradScaler |
| Batch | 4 × accum 2 (effective 8), dual forward on ref batches |

## Dev sets (verified)

- Canonical Stage A paired Dev: **n=34** (`lolv2_real_paired_dev.json` / `unified_real_dev_noref.json`)
- Correct Nano Ref Dev: **n=30** (`lolv2_real_triplet_dev.json` / `unified_real_dev_ref.json`)
- Paired-only (no Nano): low00282, low00319, low00328, low00500

## Pre-launch checks

- GPU free (0 MiB used on 24GB device) before launch.
- Unit + CUDA smoke passed (see `ASRT_V4_IMPLEMENTATION_REPORT.md`).
- Existing A/B/v3 experiment directories untouched.

## Status

| Run | Steps | Best Dev PSNR (ref) | Notes |
|---|---|---|---|
| smoke | 40 | **21.57** SSIM 0.782 (n=30) | path OK; peak mem ~7.3 GB |
| full `stage_b_v4` | **running** (PID live) | pending | launched after smoke; ~11 GB / 96% util @ step 50 |

Early full-run logs (@50): `C_T` mean≈0.10, `C_L` mean≈0.36 (independent); texture residual norms leaving zero; `loss_ref`≠`loss_noref`. Full Dev comparison vs Stage A / B-v2 deferred until checkpoints mature — not claimed yet.

## Planned comparisons (same Dev IDs)

1. Stage A Low-only  
2. Stage B-v2 Correct Nano Ref  
3. ASRT-v4 No-reference  
4. ASRT-v4 Correct Nano Ref  

Metrics: PSNR/SSIM, Ref−NoRef gain, Ref−StageA gain, per-image ΔPSNR, negative-gain rate, qualitative dark-flat mottling vs B-v2.

**Rule:** no hyperparameter tuning or checkpoint selection on official LOLv2-Real Test.
