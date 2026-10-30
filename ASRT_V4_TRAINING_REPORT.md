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
| smoke (pre-fix) | 40 | 21.57 (n=30) | obsolete — illum domain bug |
| full `stage_b_v4` | ~800 then **stopped** | no mid ckpt | **discard**; do not resume |
| smoke `stage_b_v4_fixed_smoke` | 60 | **21.94** SSIM 0.783 (n=30) | post-fix; `no_ref_prob=0.15` |
| full `stage_b_v4_fixed` | not started here | — | start via config below |

Post-review training must use `configs/stage_b_v4_fixed.yaml` (fresh Stage A warm-start). See `ASRT_V4_FIX_REPORT.md`.

## Planned comparisons (same Dev IDs)

1. Stage A Low-only  
2. Stage B-v2 Correct Nano Ref  
3. ASRT-v4 No-reference  
4. ASRT-v4 Correct Nano Ref  

Metrics: PSNR/SSIM, Ref−NoRef gain, Ref−StageA gain, per-image ΔPSNR, negative-gain rate, qualitative dark-flat mottling vs B-v2.

**Rule:** no hyperparameter tuning or checkpoint selection on official LOLv2-Real Test.
