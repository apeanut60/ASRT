# ASRT vNext V1 Implementation Report

**Project:** ASRT  
**Version:** vNext V1 — Illumination–Reference Prior Modulated Restoration  
**Date:** 2026-11-04  
**Repo:** `/root/projects/ASRT`  
**Status:** Architecture + unit tests + smoke **COMPLETE**. Formal training **NOT STARTED**.

---

## 1. New network structure

```text
Low X [-1,1]
  ├─ IlluminationEstimator (legacy light-up base) → base RGB
  ├─ IlluminationPriorEncoder → P_I {h4,h2,h1}  [B,64,·]
  └─ SFE → e0 [B,64,H] → down1 e1 [B,64,H/2] → down2 e2 [B,64,H/4]

Nano/Low Ref (A2/A3)
  └─ ReferencePriorEncoder(Z_R=[HP,Sobel_R,Sobel_X]) → P_R {h2,h1}

Coarse H/4: RB11 → coarse_irp(illum-only)
Middle H/2: skip+CSFI2 → RB21/22 → middle_irp(dual)
Fine H:     skip+CSFI3 → RB31/32/33 → fine_irp(dual)
MergeTail → residual + base → Enhanced RGB
```

Classes: `IRPCSFIUNet`, `ASRTIRP` (`build_asrt_irp`).

## 2. Diff vs old CSFIUNet

| Item | Stage A CSFIUNet | IRP v1 |
|------|------------------|--------|
| Attention | IGAB / IG_MSA (channel) | Window spatial MHSA |
| Illum modulation | Value×illum inside IG_MSA | Output modulation \(\phi_I(P_I)\) |
| Reference | TTSR / adapters (legacy paths) | Structural prior → tanh gate on SA_R |
| Fine H | ResBlocks only | + `fine_irp` dual-prior |
| Init | Often Stage-A ckpt | All scratch; no Stage-A load |
| Training | Often frozen backbone + adapter | Full E2E joint (when started) |

Old `asrt/models/csfi_unet.py`, `attention.py`, Stage A scripts untouched as runtime paths.

## 3. IRP-MSA formulas

\[
\bar F=\mathrm{Norm}(F),\quad
M_I=\mathrm{SA}_I(\bar F),\quad
M_R=\mathrm{SA}_R(\bar F)
\]

\[
Z_I=M_I\odot\phi_I(P_I),\quad
Z_R=M_R\odot\tanh(\phi_R(P_R)),\quad
M_E=\alpha Z_I+\beta Z_R
\]

\[
F'=F+W_o(M_E),\quad
F_{\mathrm{out}}=F'+\mathrm{FFN}(\mathrm{Norm}(F'))
\]

Window SA: \(A=\mathrm{softmax}(QK^\top/\sqrt{d})\), \(M=AV\); padding keys masked to \(-\infty\).

## 4. Two independent Self-Attentions

`IRPTransformerBlock` owns `attn_illum` and (if enabled) `attn_ref`, each a full `WindowSelfAttention` with private `q/k/v/proj` Linear layers. Parameter ID sets are disjoint (unit test PASS).

Q/K/V always from restoration feature — Reference never supplies K/V.

## 5. Tensor shapes (n_feats=64, input H×W)

| Stage | Feature | Illum prior | Ref prior |
|-------|---------|-------------|-----------|
| H/4 coarse | `[B,64,H/4,W/4]` | `[B,64,H/4,W/4]` | — |
| H/2 middle | `[B,64,H/2,W/2]` | `[B,64,H/2,W/2]` | `[B,64,H/2,W/2]` |
| H fine | `[B,64,H,W]` | `[B,64,H,W]` | `[B,64,H,W]` |
| Output | `[B,3,H,W]` | | |

## 6. Illumination Prior

```text
X=(low+1)*0.5;  P_I=1-X.amax(dim=1) → [B,1,H,W]
proj_h{1,2,4}: independent Conv3×3 stacks → 64-ch
```

Source: Low only. **PASS** (definition + encoder scale tests).

## 7. Reference Prior

```text
σ=7 fixed Gaussian HP; Sobel magnitude per-channel
Z_R=cat[H_R,G_R,G_X] [B,9,H,W]
stem 9→32→64 → to_h1; down → h2
```

Reuses `asrt.evaluation.reference_information.high_pass`. No GT. **PASS**.

## 8. Parameter distribution (A3 / `prior_source=nano`)

| Module | Params |
|--------|-------:|
| **Total / trainable** | **3,847,371** |
| IlluminationPriorEncoder | 112,704 |
| ReferencePriorEncoder | 131,904 |
| coarse_irp (illum-only) | 60,865 |
| middle_irp (dual) | 81,666 |
| fine_irp (dual) | 81,666 |
| IlluminationEstimator | 2,179 |
| SFE | 629,568 |
| CSFI2+RB2 path | ~853k |
| CSFI3+RB3 path | ~1.05M |
| MergeTail | 129,219 |

A1 (`none`): **3,673,865** (no ref encoder / no SA_R).  
A2 = A3 param count: **3,847,371** (**PASS**).

## 9. A1 / A2 / A3 consistency

| Check | Status |
|-------|--------|
| A2/A3 param count equal | **PASS** |
| A3(ref=low) ≡ A2 same weights | **PASS** |
| A1 fewer params / no attn_ref | **PASS** |
| Same train API (MSE) | **PASS** |

## 10. NoRef / Wrong / Shuffle

| Check | Status |
|-------|--------|
| `ref=None` skips ReferencePriorEncoder | **PASS** |
| A3 output depends on Ref (max‖Δ‖>1e-8) | **PASS** (init Δ≈1.7e-6) |
| A2 ignores external Ref | **PASS** |
| `ref_mode` Correct/Wrong/Shuffle/Ref=Low/NoRef | **PASS** (wired in `asrt/data/reference_modes.py` + `train_irp.py`) |
| Illumination always from Low | **PASS** (by construction) |

## 11. Gradient flow

| Target | Status |
|--------|--------|
| SA_I / SA_R QKV | **PASS** |
| Illum / Ref prior encoders & projections | **PASS** |
| α / β | **PASS** |
| Middle / Fine IRP | **PASS** |
| SFE / MergeTail | **PASS** |

## 12. Smoke Test

Device: **NVIDIA GeForce RTX 4090**. Artifact: `/root/data/experiments/ASRT/irp_v1_smoke/smoke_report.json`.

| Item | Result | Status |
|------|--------|--------|
| A1/A2/A3 forward+backward | OK | **PASS** |
| Tiny overfit 30 steps (B=2, 128², lr=1e-3) | 0.668 → 0.523 | **PASS** |
| AMP autocast+GradScaler | loss finite | **PASS** |
| Old ASRT `build_model(stage=a)` | runs | **PASS** |

Note: overfit curve has transient spikes (e.g. step 4≈18.5) then recovers; overall descent holds. Not used as research PSNR evidence.

## 13. GPU memory & latency (A3, batch=1)

| Size | Fwd (s) | Fwd+Bwd (s) | Peak fwd MB | Peak bwd MB |
|------|--------:|------------:|------------:|------------:|
| 128×128 | 0.028 | 0.550 | 114 | 575 |
| 256×256 | 0.034 | 0.092 | 382 | 2246 |

Window attention at full H dominates activation memory; no abnormal OOM on 4090 for these sizes.

## 14. Known engineering limits

1. At random init with \(\beta=0.1\), RGB sensitivity to Ref is tiny (~1e-6); grows with training.
2. Tiny-overfit used synthetic rand tensors (not LOLv2) for speed.
3. Formal LR / steps / splits **not frozen**; `max_steps: null` in config.
4. `train_irp.py` refuses to run without `--smoke_steps` or `--max_steps`.
5. Non-multiple-of-4 inputs are reflect/replicate-padded then cropped (same family as CSFI).
6. No MoE / Cross-Attn / reliability gate / perceptual loss by design.

## 15. ISALux alignment vs engineering deltas

**Aligned:** dual prior attention modulation; illumination from max-channel darkness map; independent illum vs structure branches; restore backbone remains U-Net-like.

**Engineering deltas (explicit):** window spatial MHSA (not claimed ISALux line-by-line); Q/K/V from restore features only (no ref cross-attn); CSFI cross-scale topology retained; structural ref prior via HP+Sobel encoder rather than ISALux’s original prior extractors; learnable α/β fusion scalars.

## 16. Git Commit

See commit message after this report is included in the tree (hash filled at commit time).

## 17. Reproduce

```bash
cd /root/projects/ASRT
PYTHONPATH=. python tests/test_irp_attention.py
PYTHONPATH=. python tests/test_irp_prior_encoder.py
PYTHONPATH=. python tests/test_irp_model_shapes.py
PYTHONPATH=. python tests/test_irp_gradient_flow.py
PYTHONPATH=. python tests/test_irp_reference_conditioning.py
PYTHONPATH=. python tests/test_irp_source_equivalence.py
PYTHONPATH=. python scripts/smoke_irp.py --config configs/asrt_irp_v1.yaml --device cuda

# Training entry only (formal schedule not started):
PYTHONPATH=. python scripts/train_irp.py --config configs/asrt_irp_v1.yaml --smoke_steps 20
# PYTHONPATH=. python scripts/train_irp.py --max_steps …   # after protocol freeze
```

## Unit test matrix (checklist §12)

| # | Item | Status |
|---|------|--------|
| 1–8 | Attention independence, shapes, mask, FFN, no IG_MSA/ref-KV | **PASS** |
| 9–15 | Prior definition, HP/Sobel, scales, no GT, shared encoder | **PASS** |
| 16–24 | H/4 / H/2 / H, 128/256/non-4×, finite, AMP | **PASS** |
| 25–32 | Gradients on SA/priors/αβ/backbone | **PASS** |
| 33–40 | Ref conditioning, NoRef, A2/A3 iso, legacy ASRT | **PASS** |

Formal 80k-step training / hyperparameter search: **NOT RUN**.

## Next stage (formal training — proposed, not frozen)

```yaml
training:
  optimizer: adamw
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  loss: mse
  batch_size: 8
  crop_size: 128
  amp: true
  # propose after smoke: e.g. max_steps / epochs on LOLv2-Real Train triplets
  prior_source: nano
  ref_mode: correct
  load_stage_a_checkpoint: false
```

Freeze LR schedule, step budget, eval protocol, and checkpointing in a follow-up task before launching full train.
