# ASRT vNext V1 — Final Protocol Fix Report

**Branch:** `vnext-v1-irp`  
**Review baseline:** `e5cb68015d643e1f107fcf21c53bfc3f645b206b`  
**Protocol version:** `irp_v1_train_v3`  
**Date:** 2026-11-04  

---

## 1. Git commit & modified files

**Commit:** `c4701eb44136e63de8accae7aa7b981488c86e71`

**Files changed:**

| Path | Change |
|------|--------|
| `asrt/data/halo_crop.py` | Outer crop origin aligned to `window_size*4`; halo ceil to alignment (32) |
| `asrt/training/irp_trainer.py` | `ref_mode` + `validation_protocol` in train protocol; formal `ref_mode=correct` gate; validate-before-mutate on resume |
| `scripts/train_irp.py` | Formal CLI rejects non-`correct` `--ref_mode`; smoke allows diagnostic modes |
| `configs/asrt_irp_v1.yaml` | `protocol_version: irp_v1_train_v3` |
| `tests/irp_test_fixtures.py` | **New** — synthetic Low/GT/Ref triplets + manifests |
| `tests/test_irp_exact_resume.py` | Fail-fast + synthetic; CPU accum1/2 + CUDA AMP |
| `tests/test_irp_a2_a3_sequence.py` | Full `state_dict` equality + sample/aug/LR |
| `tests/test_irp_config_consistency.py` | Protocol binding matrix (ref_mode / val / manifests) |
| `tests/test_irp_halo_crop.py` | Multi-scale alignment + HP/Sobel/forward tests |
| `tests/test_irp_data_reproducibility.py` | Synthetic fixtures (no external-data soft SKIP) |
| `tests/test_irp_source_equivalence.py` | Narrow exception handling (no bare `except Exception`) |

**Not modified (as required):** `IRPTransformerBlock`, `WindowSelfAttention`, `IRPCSFIUNet` structure, Illumination/Reference Prior math, A1/A2/A3 model defs, MSE loss.

**Preserved:** optimizer-update step counting, warmup+cosine, `ResumableBatchSampler`, grad accum, AMP, full-image val, checkpoint save/resume.

---

## 2. Fail-fast test repair

- Removed soft `try/except Exception: return` / print-SKIP-as-PASS patterns from Exact Resume, A2/A3 Sequence, Config Consistency.
- Unit tests that previously depended on LOLv2 paths now use `tests/irp_test_fixtures.make_synthetic_triplet_dataset` (`tmp` RGB Low/GT/Ref, non-overlapping train/val IDs, size ≥ crop+halo, Ref ≠ GT).
- CUDA-only tests use `pytest.skip()` when CUDA is absent; when CUDA is present, failures are FAIL.
- `test_irp_data_reproducibility.py` rewritten onto synthetic data (no silent PASS on missing data).

---

## 3. CPU Exact Resume

| Case | Result |
|------|--------|
| AMP=False, workers=0, accum=1 | **PASS** (param/loss atol `1e-6`) |
| AMP=False, workers=0, accum=2 | **PASS** |

Checked: sample IDs, aug metadata, LR sequence, global_step, params, optimizer, scheduler, losses.

---

## 4. CUDA AMP Resume

| Case | Result |
|------|--------|
| AMP=True, workers=2 | **PASS** (param atol threshold `1e-3`; exceed ⇒ FAIL, no PASS_SOFT) |

---

## 5. A2/A3 init & data order

- Same seed → **all** corresponding `state_dict` keys: `torch.equal` (shapes + values), not partial match counts.
- Same seed/manifest/crop/budget → A2 sample IDs == A3; aug meta ==; LR schedule ==; max_steps/batch/accum equal.
- Real-data smoke order check (3 updates, batch=4): **ids_match / aug_match / lr_match = True**.

---

## 6. Reference Mode checkpoint binding

- `train_protocol["ref_mode"]` recorded and compared on resume.
- Formal training / `IRPTrainer` default: `data.ref_mode` must be `correct` unless `allow_non_correct_ref` (smoke/diagnostic).
- CLI: `python scripts/train_irp.py --ref_mode wrong` (without `--smoke_steps`) → **SystemExit** before training.
- A1 ignores ref branch; A2 prior source = Low; A3 = Nano Ref — `ref_mode` does not substitute model `prior_source`.

---

## 7. Validation Protocol checkpoint binding

`validation_protocol` (+ stable hash) includes:

- `evaluation.inference_mode`, `irp_allow_tiled`, `irp_oom_policy`, `input_range`, `tile`, `overlap`, `full_max_side`
- `training.val_every`, `training.val_max_images`

Resume tests:

| Mutation | Expected | Result |
|----------|----------|--------|
| Same config | PASS | PASS |
| correct → ref_low | FAIL | PASS (raises) |
| correct → wrong | FAIL | PASS (raises) |
| full → tiled | FAIL | PASS (raises) |
| input_range changed | FAIL | PASS (raises) |
| val_max_images changed | FAIL | PASS (raises) |
| val_every changed | FAIL | PASS (raises) |
| train/val manifest changed | FAIL | PASS (raises) |

Protocol check runs **before** mutating trainer weights/optimizer (`load_checkpoint`). Weights-only remains a separate path.

---

## 8. Halo / window-grid alignment

**Derivation (current IRP: window=8 at H, H/2, H/4):**  
`alignment = window_size * 4 = 32` so outer crop origin is a common lattice under exact /2,/4 maps.

**Implementation:**

- `required_prior_context = gaussian_radius(σ=7)+sobel = 22`
- `halo = ceil(22/32)*32 = 32` via `compute_irp_halo`
- `outer_top/left % 32 == 0`; `valid_* = outer_* + halo` (also aligned)

**Caveat (not bit-exact claimed):** CSFI stride-2 Conv uses `padding=1`; Full vs crop window tokens are phase-aligned at the outer lattice but not guaranteed bit-identical to full-image after network downsampling. Flip/rot after crop remaps local windows; tests verify sync geometry across Low/GT/Ref and finite forward/backward.

Halo 22→32 is a **context/crop policy** change, not a Gaussian/Prior math change. A2/A3 share the same halo.

---

## 9. Full IRP test inventory

```text
python -m pytest tests/test_irp_*.py
→ 68 passed, 0 failed, 0 skipped, 0 blocked
```

| Suite | PASS | FAIL | SKIPPED | BLOCKED |
|-------|------|------|---------|---------|
| `test_irp_exact_resume` | 3 | 0 | 0 | 0 |
| `test_irp_a2_a3_sequence` | 2 | 0 | 0 | 0 |
| `test_irp_config_consistency` | 3 | 0 | 0 | 0 |
| `test_irp_halo_crop` | 9 | 0 | 0 | 0 |
| `test_irp_scheduler_steps` | 5 | 0 | 0 | 0 |
| `test_irp_sampler_resume` | 3 | 0 | 0 | 0 |
| `test_irp_accum_equivalence` | 1 | 0 | 0 | 0 |
| `test_irp_checkpoint_protocol` | 6 | 0 | 0 | 0 |
| Other `test_irp_*` | 36 | 0 | 0 | 0 |
| **Total** | **68** | **0** | **0** | **0** |

Critical CPU suites (Exact Resume / A2-A3 / Config Consistency) do **not** skip for missing LOLv2.

---

## 10. Real Triplet short smoke

**Dir:** `/root/data/experiments/ASRT/irp_v1_final_protocol_smoke`  
**Status:** **PASS** (engineering preflight only — **not** Nano-Ref effectiveness / paper evidence)

| Check | A2 (`prior=low`) | A3 (`prior=nano`) |
|-------|------------------|-------------------|
| 100 optimizer updates | PASS | PASS |
| Interrupt @50 + resume → 100 | PASS | PASS |
| Validation (full, n=2) | PASS | PASS |
| Halo | 32 | 32 |
| ref_mode | correct | correct |
| Params | 3,847,371 | 3,847,371 |
| NaN/Inf | none | none |
| A2/A3 sample order (3 steps) | match | match |

PSNR at 100 steps is **not** used to judge Nano Ref.

---

## 11. Remaining engineering limits

1. Full-image vs halo-crop window tokens are lattice-aligned, not claimed bit-exact through CSFI padding.
2. CUDA AMP resume uses finite atol (`1e-3`); CPU uses strict `1e-6`.
3. Formal `max_steps` for 80k is still unset in config (`null`) — must be set explicitly to launch.
4. Diagnostic Wrong / Shuffle / Ref=Low / NoRef remain eval interventions only.
5. Tiled IRP inference remains experimental (`irp_allow_tiled=false` for formal PSNR).

---

## 12. Formal training READY?

**READY = YES** for launching formal A1/A2/A3 controlled runs under:

- `data.ref_mode: correct`
- `protocol_version: irp_v1_train_v3`
- Auto halo = 32
- Full-image validation
- Explicit `--max_steps 80000` (or config)

Critical regression tests all **PASS**; no critical FAIL remains.
Do **not** treat the 100-step smoke as performance evidence.
