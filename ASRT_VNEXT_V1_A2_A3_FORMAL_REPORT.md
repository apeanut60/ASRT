# ASRT vNext V1 — Formal A2/A3 Training Report

**Status:** `COMPLETE` (A2 + A3 80k finished)  
**Code freeze for runs:** `a48c316c9644f83c2f565dc85e59b62a8eb73c34` (`num_workers=4`)  
**Protocol:** `irp_v1_train_v3`  
**Updated:** 2026-11-05

> Dev validation (32 images) + single seed. **Not** an independent paper test set.  
> See also gate-collapse diagnostics: `ASRT_VNEXT_V1_GATE_COLLAPSE_DIAG_REPORT.md`.

---

## 1. Git / identity

| Item | Value |
|------|-------|
| A2/A3 `resolved_config.git_commit` | `a48c316…` |
| `git_dirty` | false |
| Prior sources | A2=`low`, A3=`nano` |
| Train/val manifests | `lolv2_real_triplet_{train,dev}.json` (607 / 32, no ID overlap) |

Incomplete `workers=0` A2 (stopped @4850) archived under  
`vnext_v1/archive_incomplete_a2_workers0_step4850/` — **not** used for results.

---

## 2. Training recipe (frozen)

Scratch init, Adam 1e-4, wd 1e-4, warmup 1000 + cosine, MSE, crop 128, halo 32 (input 192), batch 8, AMP, seed 42, `ref_mode=correct`, full-image val every 2000.

Wall clock (workers=4): A2 ≈ 8.1 h, A3 ≈ 7.7 h.

---

## 3–5. Data / init / preflight

Preflight PASS @ batch 8; A2/A3 init weights equal @ seed 42 (3,847,371 params).  
Fingerprints matched. Formal runs re-initialized from seed 42 (preflight ckpts unused).

---

## 6–11. Results from disk logs

| | A2 | A3 |
|--|--|--|
| Final step | 80000 | 80000 |
| Best val PSNR | **17.487** @62k | **17.675** @66k |
| Best val SSIM | 0.7072 | **0.7260** |
| Last @80k PSNR | 17.364 | 17.565 |
| Last @80k SSIM | 0.7050 | 0.7251 |
| Best ckpt | `.../A2_.../best_validation_psnr.pt` | `.../A3_.../best_validation_psnr.pt` |

**Best-vs-best (image-paired on 32 val images):** mean ΔPSNR (A3−A2) = **+0.187 dB**,  
bootstrap 95% CI over **images** = **[−0.092, +0.461]** (includes 0; 62.5% images favor A3).

**Do not** bootstrap the 40 training-time val checkpoints as if independent trials.

### Gate collapse (log fact)

| Gate | |gate|&lt;1e-3 | |gate|&lt;1e-6 |
|------|-----------|-----------|
| β (h2/h1) | ~3.0k | ~4.3k |
| α (h4/h2/h1) | ~15.1k | ~16.8k |

Post-collapse, A3 best/last checkpoints show **zero** float-output sensitivity to Correct/Wrong/NoRef (MAE=0); see gate diag report.

---

## 12–16. Diagnostics / viz / limits

- Reference interventions + branch contribution: **DONE** under `irp_gate_diagnostics/`.
- Forced gate ablations B0/B1/B2 (10k): see gate diag report (may still be running).
- Limits: single seed; tiny absolute gap; **dev set reused in engineering**; gates die early → Nano Prior not shown to be used at final ckpt.

---

## 17. Follow-ups

1. Finish / interpret B0–B2 gate WD/LR ablations.  
2. Only if gates stay alive **and** Ref sensitivity is beneficial → consider full V2 80k multi-seed.  
3. Independent holdout / official test before any Nano-effectiveness claim.
