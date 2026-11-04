# ASRT vNext V1 Repair Report

**Branch:** `vnext-v1-irp`  
**Base HEAD (pre-repair):** `62051173d45138dc49d2cb5a2e9bf1ffea5827cd`  
**Date:** 2026-11-04  
**Scope:** Implementation / experiment-protocol repair only — **no** network redesign.

---

## 1. Fix summary (P0 / P1 / P2)

### P0-1 — Missing `reference_information` dependency

| | |
|--|--|
| **Problem** | `reference_prior_encoder.py` imported `asrt.evaluation.reference_information.high_pass`, absent on this branch → clean import failure. |
| **Root cause** | Diagnostics module lived only on later local `v4` commits (R1.x), not cherry-picked into `vnext-v1-irp`. |
| **Files** | `asrt/models/prior_ops.py` (**new**); `asrt/models/reference_prior_encoder.py` |
| **Fix** | Model-local separable Gaussian blur + high-pass (ported from historical R1.x: radius=`ceil(3σ)`, reflect pad, dtype-preserving). |
| **Verify** | `python -c "from asrt.models.asrt_irp import ASRTIRP"` **PASS**; prior unit tests **PASS**. |

### P0-2 — Tiled inference kwargs / Full–Tiled protocol

| | |
|--|--|
| **Problem** | `infer_tiled` always passed `query_gamma` / `query_lut` (TTSR); IRP `forward` rejects them. |
| **Root cause** | Single call site assumed legacy/v4 signature. |
| **Files** | `asrt/evaluation/inference.py`, `asrt/evaluation/protocol.py`; `tests/test_irp_inference_protocol.py` |
| **Fix** | Explicit `is_irp_model` / `call_model` dispatch (no silent `TypeError` retry). IRP tiled: window-grid–aligned starts + HP/Sobel/window halo. Formal protocol **forces full-image** unless `evaluation.irp_allow_tiled=true`; OOM does not silently fall back to tiled. |
| **Verify** | Unit + GPU: uncovered=0; Full vs Tiled **not bit-exact** (see §3). Legacy tiled path still runs. **PASS** |

### P1-1 — Checkpoint family / `prior_source` consistency

| | |
|--|--|
| **Problem** | `detect_ckpt_version` missed IRP; A2/A3 identical weight shapes → silent wrong `prior_source` load risk. |
| **Files** | `asrt/models/factory.py`, `asrt/models/asrt_irp.py`; `tests/test_irp_checkpoint_protocol.py` |
| **Fix** | `model_family=asrt_irp`, `model_version=irp_v1`, mandatory `irp_hparams` in ckpt; strict compare incl. `prior_source`, window, σ, α/β inits. Override only via `allow_hparam_override` (logged). |
| **Verify** | A3→A3 PASS; A3→A2 FAIL; window/σ mismatch FAIL; missing meta FAIL. **PASS** |

### P1-2 — Halo crop for Reference HP boundaries

| | |
|--|--|
| **Problem** | Crop-then-HP creates false border high-frequency. |
| **Files** | `asrt/data/halo_crop.py`, `asrt/data/irp_dataset.py`; `tests/test_irp_halo_crop.py` |
| **Fix** | Sync crop coords on valid region; extract `crop+2*halo` context (`halo=prior_context_halo(σ=7)+sobel≈22`); loss on center. Optional snap of valid top/left to window grid. |
| **Verify** | Center HP/Sobel vs full-image max abs \< 1e-4 (interior). **PASS** |

### P1-3 — Wrong / Shuffle diagnostics

| | |
|--|--|
| **Problem** | BS=1 Wrong → horizontal flip; only pixel shuffle. |
| **Files** | `asrt/data/reference_modes.py`; `tests/test_irp_reference_modes.py` |
| **Fix** | Fixed Wrong mapping JSON (no self; fail if missing). Modes: `wrong`, `patch_shuffle`, `pixel_shuffle` (legacy `shuffle`→pixel), `ref_low`, `noref`. Training default remains **Correct**. |
| **Verify** | Mapping + fail-closed Wrong. **PASS** |

### P1-4 — Formal training infrastructure

| | |
|--|--|
| **Problem** | `train_irp.py` was smoke-only. |
| **Files** | `asrt/training/irp_trainer.py`, `scripts/train_irp.py`, `configs/asrt_irp_v1.yaml`; resume/accum tests |
| **Fix** | `IRPTrainer`: Adam/AdamW, warmup+cosine, clip, AMP, true grad accum (`loss/accum` then step), rich checkpoints (model/opt/sched/scaler/step/epoch/RNG/DL generator/manifest hashes/`irp_hparams`), `--resume`, val via full-image protocol, `best_validation_psnr` / `last` / periodic. Dataset `set_epoch` on loader restart. |
| **Verify** | Short formal run + resume step continuity **PASS**. Full 80k **NOT RUN**. |

### P1-5 — Tiny overfit stability audit

| | |
|--|--|
| **Problem** | Prior report: synth overfit spike ≈18.5 at smoke `lr=1e-3`. |
| **Files** | `scripts/smoke_irp_overfit_audit.py`; artifact `.../irp_v1_repair/overfit_audit/` |
| **Fix** | No architecture change. Reproduced + compared candidate formal LR. |
| **Result** | Synth `lr=1e-3`: spike **yes** (max≈18.7). Synth `lr=1e-4` (±AMP): **no spike**, monotone drop. Real triplet `lr=1e-4`: strong descent 0.66→0.18; transient max≈2.5 (mild). **Do not redesign IRP-MSA.** Prefer formal candidate `lr=1e-4` + clip. |

### P2 — Config / protocol asserts

| | |
|--|--|
| **Files** | `validate_irp_cfg` in `irp_trainer.py`; model ctor checks heads÷channels, prior_source; AMP-without-CUDA refused; nano missing-ref refused for Correct train. |
| **Verify** | Exercised via trainer init + unit tests. **PASS** |

---

## 2. Dependency integrity

Clean import without evaluation diagnostics package:

```bash
PYTHONPATH=. python -c "from asrt.models.asrt_irp import ASRTIRP"
PYTHONPATH=. python -c "from asrt.models.reference_prior_encoder import ReferencePriorEncoder"
```

**PASS** on this tree. Gaussian ops live in `asrt/models/prior_ops.py` and are committed.

Numerical note: implementation follows historical R1.x separable reflect Gaussian (`truncate=3σ`); bit-identity vs deleted eval module was not re-proven on this branch (file absent).

---

## 3. Full / Tiled evaluation

| Setting | Result | Status |
|---------|--------|--------|
| IRP `call_model` without query kwargs | OK | **PASS** |
| Formal `inference_mode=tiled` without `irp_allow_tiled` | hard error | **PASS** |
| Experimental tiled uncovered pixels | 0 | **PASS** |
| Full vs Tiled 256² (tile=128, overlap=32, halo=46) | max‖Δ‖≈2.2e-3, mean≈9.5e-6, RMSE≈4.2e-5 | **PASS** (quantified; **not** claimed equal) |
| Legacy Stage-A full/tiled | OK | **PASS** |

**Policy:** Formal PSNR uses **full-image**. Tiled is diagnostic / experimental only.

---

## 4. A1 / A2 / A3 consistency

| Check | Result | Status |
|-------|--------|--------|
| Params A2 == A3 | 3,847,371 | **PASS** |
| A1 params | 3,673,865 | **PASS** |
| Same train entry / MSE | yes | **PASS** |
| A3 interventions keep `prior_source=nano` | enforced by ckpt loader | **PASS** |
| Shared aug seed/epoch geometry | data repro tests | **PASS** |

---

## 5. Training readiness

| Capability | Status |
|------------|--------|
| Smoke `--smoke_steps` | **PASS** |
| Formal `IRPTrainer` (opt/sched/AMP/accum/ckpt/resume/val) | **PASS** (short) |
| Halo crop + center loss | **PASS** |
| Wrong mapping artifact | auto-built under `out_dir` | **PASS** |
| 80k-step run | **NOT RUN** |
| Final LR / step budget freeze | **NOT RUN** (config candidates only) |

Recommended entry (after freezing `max_steps`):

```bash
PYTHONPATH=. python scripts/train_irp.py \
  --config configs/asrt_irp_v1.yaml \
  --max_steps 80000
# resume:
PYTHONPATH=. python scripts/train_irp.py --config configs/asrt_irp_v1.yaml \
  --resume /path/to/checkpoints/last.pt --max_steps 80000
```

---

## 6. Regression

| Check | Status |
|-------|--------|
| Existing IRP unit suite (6 files) | **PASS** |
| Legacy `build_model(stage=a)` + `infer_full/tiled` | **PASS** |
| V3/V4 full train regression suite | **NOT RUN** (inference path unchanged for non-IRP) |

---

## 7. Remaining issues

| Item | Status |
|------|--------|
| Full vs Tiled bit-exact IRP | **FAIL** by design (window/grid/feather); errors quantified |
| Multi-worker DL bit-exact resume | **BLOCKED** / limited — RNG+generator saved; worker nondeterminism may remain (`num_workers=0` for strict checks) |
| Near-duplicate exclusion in Wrong mapping | **NOT RUN** (no near-dup metadata); self-pairs forbidden only |
| Formal 80k + best-ckpt selection on val | **NOT RUN** |
| Synth `lr=1e-3` loss spike | **PASS** as documented limitation; avoid that LR |
| Unrelated working-tree doc moves (`ASRT_*.md` → `docs/`) | **untouched** by this repair |

---

## 8. Git / reproduce

### Tests run

```bash
cd /root/projects/ASRT
for t in tests/test_irp_{attention,prior_encoder,model_shapes,gradient_flow,reference_conditioning,source_equivalence,halo_crop,reference_modes,checkpoint_protocol,training_resume,data_reproducibility,inference_protocol}.py; do
  PYTHONPATH=. python "$t"
done
PYTHONPATH=. python scripts/smoke_irp_overfit_audit.py --device cuda
# GPU formal smoke artifact:
# /root/data/experiments/ASRT/irp_v1_repair/formal_smoke/gpu_smoke_report.json
```

### Modified / added files (repair)

```text
asrt/models/prior_ops.py
asrt/models/reference_prior_encoder.py
asrt/models/asrt_irp.py
asrt/models/factory.py
asrt/evaluation/inference.py
asrt/evaluation/protocol.py
asrt/data/halo_crop.py
asrt/data/irp_dataset.py
asrt/data/reference_modes.py
asrt/training/irp_trainer.py
configs/asrt_irp_v1.yaml
scripts/train_irp.py
scripts/smoke_irp_overfit_audit.py
tests/test_irp_inference_protocol.py
tests/test_irp_checkpoint_protocol.py
tests/test_irp_halo_crop.py
tests/test_irp_reference_modes.py
tests/test_irp_training_resume.py
tests/test_irp_data_reproducibility.py
ASRT_VNEXT_V1_REPAIR_REPORT.md
```

### Commit

**Hash:** `f7e86eda67527b226ec811d35a52f656e449ef73`

### Blockers before claiming Reference Gain

- No formal train / val selection yet.  
- Full/Tiled residual error must not be ignored if tiled PSNR is ever published.  
- Overfit / short-run loss ≠ evidence of useful Nano conditioning.
