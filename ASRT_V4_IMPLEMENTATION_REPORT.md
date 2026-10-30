# ASRT-v4 Implementation Report

## Code added / touched

### New files
- `asrt/models/v4_ref_encoder.py` — lightweight Ref encoder
- `asrt/models/v4_matching.py` — local soft matcher + warp helpers
- `asrt/models/v4_reliability.py` — dual C_T / C_L estimator
- `asrt/models/v4_adapters.py` — texture / illum adapters + LPF luminance
- `asrt/models/asrt_v4.py` — top-level ASRTv4 + Stage A load report
- `configs/stage_b_v4.yaml`
- `tests/test_asrt_v4.py`
- Docs: `ASRT_V4_*.md`

### Modified (backward compatible)
- `asrt/models/csfi_unet.py` — `enable_v4`, `encode_low_stem`, texture/illum adapter path; v3 path unchanged
- `asrt/training/optimizer.py` — `v4_new` param group
- `asrt/training/trainer.py` — build ASRTv4, Stage A warm-start, dual-forward L1
- `asrt/evaluation/inference.py` — matching halo for tiled v4
- `scripts/train.py` — stage `b_v4` / `v4` uses B-v2 manifests
- `asrt/models/__init__.py` — export ASRTv4

### Not modified (preserved)
- Stage A/B/v2/v3 configs, experiment dirs, LTE/SearchTransfer, CondResBlock, IGAB internals, ResBlock math.

## Audit summary (pre-change)

**Stage A forward (actual):**
`Low[-1,1] → pad÷4 → IllumEstimator(f∈R^{64}, log_gain) → light-up base → SFE(e0) → down(e1 H/2) → down(e2 H/4) → inject heads (optional S/T) → RB11+IGAB(e2, L_h4) → CSFI2 → RB/IGAB middle → CSFI3 → RB fine → MergeTail residual + base → crop`

**IGAB:** channel-transposed attention; `V_mod = V * illu`; Q/K from restore features. Unchanged in v4.

**Stage A best:** step 58000, PSNR≈21.43 / SSIM≈0.772 on paired Dev (n=32 in ckpt meta). Keys: `backbone.*` (230) + `lte.*` (12).

**B-v2 clean Dev:** paired n=34; correct Nano Ref triplet n=30 (missing Nano: low00282, low00319, low00328, low00500).

## Stage A → v4 load report (smoke)

```
loaded: 230
expected_new: 89   # ref_encoder, matcher, reliability, adapters, …
unused_ckpt: 12    # lte.* (intentionally unused)
shape_mismatch: 0
```

No backbone key silently dropped (`strict=False` only after filtering; missing non-v4 keys would raise).

## Acceptance tests (`tests/test_asrt_v4.py`)

| Test | Result |
|---|---|
| Local soft matcher shapes / grad / boundary mask | PASS |
| Warp non-square | PASS |
| Zero-init adapters identity | PASS |
| C_T=C_L≈0 → ref ≈ no-ref | PASS |
| `ref=None` skips RefEncoder | PASS |
| No LTE / no CondResBlock | PASS |
| E2E grads after 5 Adam steps (Q/K/RefEnc/Reli/Adapters/SFE) | PASS |
| Stage A load report | PASS |
| Optimizer `v4_new` group | PASS |
| Dual-forward L1 backward | PASS |
| CUDA smoke: 3 steps, full vs tiled | PASS (mean |Δ|≈0.013 @128×160) |

## Dual-forward training

```
L = L1(pred_ref, gt) + λ_no_ref * L1(pred_no_ref, gt)
λ_no_ref = 0.5 (config)
```

No frozen teacher. No GT exposure matching.

## Single-backbone check

- One `CSFIUNet` instance; one `MergeTail`.
- Matching features come from shared stem (`encode_low_stem`) + RefEncoder (not a second restore U-Net).
- No extra RGB refinement net.

## Detach / freeze audit

- No `detach()` on Q/K/W/disp/grid in matching path.
- No permanent `eval()` / `requires_grad=False` on trunk during train.
- Adapter zero-init: first backward may show weak adapter-pre-last grads; verified non-zero after several steps.

## Commands

```bash
# unit tests
cd /root/projects/ASRT && PYTHONUNBUFFERED=1 python tests/test_asrt_v4.py

# smoke train (e.g. 100 steps)
python scripts/train.py --config configs/stage_b_v4.yaml --max_steps 100 \
  --out_dir /root/data/experiments/ASRT/stage_b_v4_smoke

# full train
python scripts/train.py --config configs/stage_b_v4.yaml
```
