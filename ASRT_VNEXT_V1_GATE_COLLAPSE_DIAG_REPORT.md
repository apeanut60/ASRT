# ASRT vNext V1 — Gate Collapse Diagnosis Report

**Date:** 2026-11-05  
**Baseline:** `a48c316`  
**Artifacts:** `/root/data/experiments/ASRT/vnext_v1/irp_gate_diagnostics/`

---

## 1. A2/A3 formal result (limits)

| | A2 | A3 |
|--|--|--|
| Best val PSNR | 17.487 @62k | 17.675 @66k |
| Best val SSIM | 0.7072 | 0.7260 |

Best-vs-best **image-paired** ΔPSNR = +0.187 dB; bootstrap over **32 images** CI **[−0.09, +0.46]** (includes 0).  
Dev set + single seed. Do **not** bootstrap the 40 train-time val checkpoints as independent trials.

---

## 2. Checkpoint identity

- A2/A3 artifacts present; `load_checkpoint_strict` rejects A2↔A3 prior_source mismatch.
- Negative tests: `a3_into_a2` / `a2_into_a3` → **REJECTED**.
- See `checkpoint_audit.json`.

---

## 3. Gate timeline (A3 train_log)

| Gate | &lt;1e-3 | &lt;1e-6 |
|------|--------|--------|
| β h2/h1 | ~3.0k | ~4.3k |
| α h4/h2/h1 | ~15.1k | ~16.8k |

Plot: `plots/a3_gate_timeline.png`

---

## 4. Reference interventions (A3 best@66k & last@80k)

All modes Correct / Wrong / PatchShuffle / PixelShuffle / Ref=Low / NoRef:

- mean ΔPSNR vs Correct = **0**
- float MAE / RMSE / changed-pixel ratio (thresh 1e-5) = **0**

**Final A3 does not use Nano Ref at inference.**  
(Does not prove Ref never mattered early in training.)

---

## 5. Branch contribution

At best/last: `ρ_R≈0`, `β M_R` L2≈0, Ref ON vs OFF output MAE=**0**.  
Scalar gate collapse ⇒ effective reference branch inactive at these ckpts.

---

## 6. Optimizer audit

V1 Adam: **single param group**, `lr=1e-4`, `weight_decay=1e-4` — gates share WD with all weights.  
`gate_optimizer_audit.json`: `gates_share_weight_decay=true`.

---

## 7. B0 / B1 / B2 (10k updates, nano, seed 42)

| Exp | Gate LR | Gate WD | Groups | β&lt;1e-3 | Final β_h2 | Final α_h4 | Best val PSNR |
|-----|---------|---------|--------|---------|------------|------------|---------------|
| **B0** V1-equiv | 1e-4 | **1e-4** | single | **@3k** | **~0** | 0.53 | 16.05 |
| **B1** | 1e-4 | **0** | split | none | **0.099** | **1.001** | 16.12 |
| **B2** | **1e-5** | **0** | split | none | **0.100** | **1.000** | 16.00 |

Summary: `ablations/short_ablation_summary.json`

---

## 8–11. Decision

**Case A supported:** Gate **weight decay** is a major cause of β collapse.  
B0 reproduces collapse; B1 (WD=0) keeps α≈1, β≈0.1. B2 unnecessary once WD=0.

**Worth a controlled V2 80k?** Conditionally yes — only with gate WD=0 (or decoupled gate group), plus Ref-intervention checks that sensitivity stays nonzero; then multi-seed.  
10k PSNR is **not** paper evidence.

---

## Key paths

```text
/root/projects/ASRT/ASRT_VNEXT_V1_GATE_COLLAPSE_DIAG_REPORT.md   ← this file
/root/data/experiments/ASRT/vnext_v1/A2_A3_FINDINGS_FROM_LOGS.md
/root/projects/ASRT/ASRT_VNEXT_V1_A2_A3_FORMAL_REPORT.md
/root/data/experiments/ASRT/vnext_v1/irp_gate_diagnostics/
```
