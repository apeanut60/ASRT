# ASRT-v4 Code Review Fix Report

Baseline: `0e6c317` on branch `v4`.  
Old training `stage_b_v4` was **stopped** (~step 800, no mid checkpoint saved). **Not overwritten.**  
New experiment dir: `stage_b_v4_fixed` / smoke `stage_b_v4_fixed_smoke`.

## Fixes

### P0 — Reference illumination domain (`asrt_v4.py`, `v4_adapters.py`)
- **Root cause:** `LowPassLuminance` expected RGB∈[0,1] but received [-1,1].
- **Fix:** explicit `(x+1)*0.5` clamp before LPF; reflect/replicate pad (no zero-pad dark rim).
- **Verified:** equal gray → D≈0; brighter/darker Ref → D sign correct; extremes finite (FP32/AMP).

### P0 — Independent v4 eval (`factory.py`, `evaluate.py`, `export_visual_cases.py`)
- **Root cause:** scripts always `build_asrt()` + `strict=False`.
- **Fix:** `build_and_load` / `load_checkpoint_strict`; v4 via `build_asrt_v4`; refuse silent v4→legacy load.
- **Verified:** strict roundtrip; legacy refuse; Stage A evaluate↔direct forward identical PSNR.

### P1 — Metrics range (`metrics.py`)
- **Root cause:** `min()<-0.05` heuristic.
- **Fix:** required `input_range='minus_one_one'|'zero_one'`. Formal eval uses `minus_one_one`.
- **Endpoints:** -1→0, 0→128, 1→255. Identity PSNR=inf.

### P1 — Tiled inference (`inference.py`)
- Outer borders weight=1; feather only interior overlaps.
- Dynamic `tile_h/w` for H>tile&W<tile (and vice versa); no negative origins.
- Uncovered pixels raise (no `1e-8` cover-up).
- **Full/Tiled MAE** (tile=96): ≤3e-6 typical; 513×769 border MAE ~2e-5.

### P1 — Ref pad geometry (`asrt_v4.match_reference`)
- Resize Ref to **original** Low HW, then same `safe_pad` as Low (not resize-into-pad).
- Tested 255×257, 257×259, 60×85, 63×67 with ramp markers.

### P1 — No-ref sampling (`stage_b_v4.yaml`, trainer)
- `no_ref_prob: 0.15` (engineering initial, not tuned) + dual_forward retained.
- Logs `ref_batches` / `pure_noref_batches` / dual vs pure no-ref losses / Real·Syn counts.
- Smoke 60 steps: **45 ref dual + 15 pure no-ref**.

### P2 — Matching diagnostics (`v4_matching.py`)
- `best_sim` masked to valid candidates.
- Continuity via neighbor diffs (no `roll` wrap).
- Entropy / `log(n_valid)`; singleton → 0.
- Pre-L2 `q_norm_pre` / `k_norm_pre`.

### Unused inject heads
- `conv{11,22,33}_head` kept for Stage A load, **frozen** (`requires_grad=False`).
- Params: total **4,739,286**; trainable **4,370,454**; frozen inject **368,832**.

## Architecture preserved
Single CSFI backbone; trainable RefEncoder + local soft match; dual C_T/C_L; plain ResBlock; IGAB internals unchanged; no LTE / hard argmax / CondRB / teacher / second U-Net.

## Re-train recommendation
Illum-domain + `no_ref_prob` change the optimized objective → **do not continue** old `stage_b_v4`.  
Restart from Stage A @58k into `stage_b_v4_fixed`:

```bash
python scripts/train.py --config configs/stage_b_v4_fixed.yaml
```

Smoke (@60): Dev ref PSNR **21.94** / SSIM **0.783** (n=30) — init-era only, not a claim of final quality.
