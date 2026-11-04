# ASRT-v4 Architecture

**Geometry-constrained, Reliability-aware, End-to-end Reference-guided Low-Light Enhancement**

## Design constraints (satisfied)

| Constraint | Status |
|---|---|
| Single Restoration Backbone + single RGB MergeTail | Yes (`CSFIUNet` with `enable_v4=True`) |
| No Stage-A enhance → second restore | Yes |
| Stage A only as weight init | Yes (`load_stage_a_into_v4`) |
| Matching / Reliability / Adapters / Backbone jointly trained | Yes |
| No Frozen VGG19-LTE as matching encoder | Yes (no `LTE` instance) |
| No Hard Argmax / Hard Top-k primary matching | Yes (local soft softmax) |
| IGAB internals unchanged; only Illumination Guidance input | Yes |
| Plain ResBlock (no CondResBlock / FiLM) | Yes |
| `ref=None` bypass | Yes |
| Does not overwrite Stage A / B-v1 / B-v2 / v3 code or experiments | Yes (new modules + `stage_b_v4/`) |

## Data flow

```
Low X → IlluminationEstimator → SFE → downs → (e0,e1,e2), (L_h2,L_h4)
                                              │
Nano Ref R → RefEncoderV4 → F1,F2,F3
                                              │
Query = proj(e2 ∥ L_h4)  Key = proj(F3)
        └─ Local Soft Matching (r=4, 9×9) → W, disp, diagnostics
              ├─ Soft aggregate → T_h4
              └─ Expected-disp warp → T_h2, T_h1
                                              │
ReliabilityEncoder(match diag + structure) → C_texture, C_illum ∈ [0,1]
                                              │
Texture:  F' = F + C_T · Adapter(F,T)     @ H/4, H/2, H
Illum:    L' = L + C_L · Adapter(L,P,D)   @ H/4, H/2 → IGAB only
                                              │
CSFI (CSFI2/CSFI3) + ResBlocks + IGAB(F, L') + MergeTail
                                              │
Base light-up + RGB residual → Enhanced
```

## Modules

| Module | File | Trainable |
|---|---|---|
| `RefEncoderV4` | `asrt/models/v4_ref_encoder.py` | Yes |
| `LocalSoftMatcher` | `asrt/models/v4_matching.py` | Yes (Q/K proj) |
| Multiscale warp | `v4_matching.warp_by_disp` | Differentiable |
| `DualReliabilityEstimator` | `asrt/models/v4_reliability.py` | Yes |
| `TextureAdapter` / `IlluminationAdapter` | `asrt/models/v4_adapters.py` | Yes (zero-init last) |
| `ASRTv4` | `asrt/models/asrt_v4.py` | Yes |
| `CSFIUNet` (`enable_v4`) | `asrt/models/csfi_unet.py` | Shared trunk |

Legacy LTE / SearchTransfer remain in `asrt/models/reference.py` for A/B/v3 only.

## Matching

- Resolution: H/4 only.
- Window: `radius=4` → 9×9 (configurable).
- Score: `cos(Q,K)/τ − λ‖Δpos‖²_norm`, invalid pad masked before softmax.
- Complexity: `O(B · H/4 · W/4 · window · C)` via `F.unfold` (no HW×HW matrix).
- H/2 & H reuse expected displacement + `grid_sample` (`align_corners=True`).

## Parameter counts (measured)

| Model | Total params |
|---|---|
| Stage A (`ASRT`, no v3) | 4,412,928 |
| Stage B-v3 | 4,601,875 |
| **ASRT-v4** | **4,739,286** (all trainable) |

Δ vs Stage A: **+326,358** (~7.4%). Δ vs B-v3: **+137,411**.

## FLOPs / MACs (approximate)

- Protocol: sum Conv2d MACs for one forward at **256×256**, batch=1.
- **No-ref**: ≈ **90.5 G** MAC.
- **With ref**: ≈ **105.7 G** MAC (includes RefEncoder + local unfold correlation + adapters).
- Softmax / Linear IGAB ops not fully counted in the Conv-only estimate; treat as lower bound on attention cost (same IGAB as Stage A).

## Init

- Backbone: Stage A `best.pt` @ step **58000** (`/root/data/experiments/ASRT/stage_a/checkpoints/best.pt`).
- New modules: default PyTorch init; adapter last layers **zero**.
- Optimizer / scheduler / scaler / step: **not** inherited from Stage A.
