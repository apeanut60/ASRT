# Architecture

## Overview

```text
Low ──► SFE / E0–E2 ──► Coarse (H/4 + T_lv3 + IGAB)
                     │
IllumEstimator ──────┼──► Middle (H/2 + CSFI2 + T_lv2 + IGAB)
                     │
Nano Ref ─► LTE+ST ──┼──► Fine  (H   + CSFI3 + T_lv1)
                     │
                     └──► Merge RGB residual + light-up Y_base → Ŷ
```

Top-level class: `asrt.models.asrt.ASRT`.

## Forward contract

```python
out = model(low, ref=None)                 # tensor [-1, 1]
out, aux = model(low, ref=ref, return_aux=True)
```

- Inputs: RGB in `[-1, 1]`, shape `[B,3,H,W]`.
- `ref=None`: skip LTE, SearchTransfer, three inject heads, and `RefIllumTransfer`.
- Single pad to multiples of 4 before encoder/LTE; crop back to original `H×W`.
- Defaults: `n_feats=64`, `num_res_blocks='8+8+4+2'`, `use_global_illum=False`, `use_ref_illum=False`.

## Output composition

`Y_base = 2 * (Low01 * exp(log_gain)) - 1` plus RGB residual from `MergeTail`. Soft-init on inject heads keeps Stage A near identity w.r.t. reference path.

## Reference path

LTE (frozen VGG19 slices) + `SearchTransfer` produce `(S, T_lv3, T_lv2, T_lv1)` at native grids. Injectors: `feat + σ(S) * head([feat, T])`.
