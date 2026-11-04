# ASRT-v4 Evaluation Audit (post metrics / loader fix)

Protocol: RGB PSNR/SSIM, `input_range=minus_one_one`, **no** GT mean-align, full-image infer unless noted.  
Manifests: `manifests_b_v2`.

## Loader

All numbers below used `scripts/evaluate.py` → `build_and_load` + **`strict=True`**.

## Results

### Stage A No-ref — Paired Dev **n=34**
- Ckpt: `stage_a/checkpoints/best.pt` @58k  
- PSNR **22.504** / SSIM **0.778**  
- Note: ckpt meta previously cited ~21.43 on n=32; re-eval here is **n=34** + explicit range. Do not mix n without labeling.

### Stage B-v2 — Correct Nano Ref Dev **n=30**
- Ckpt: `stage_b_v2/checkpoints/best.pt` @70k  
- Bypass (no-ref): PSNR **21.565** / SSIM **0.786**  
- Correct Ref: PSNR **21.589** / SSIM **0.781**  
- Ref−Bypass ΔPSNR ≈ **+0.024** (tiny; consistent with prior B-v2 findings)

### ASRT-v4
- Old `stage_b_v4` (~800 steps): **no usable mid checkpoint** (save_every=2000); **do not** treat as a valid trained model. Illum-domain bug was active during that run → discard for comparisons.
- Fixed smoke `stage_b_v4_fixed_smoke` @60 (Trainer internal eval, ref Dev n=30): PSNR **21.945** / SSIM **0.783** — warm-start era only.
- Full fixed train / formal v4 Correct vs No-ref on n=30: **pending** `stage_b_v4_fixed` completion.

## Consistency
- Stage A, one Dev image: `infer_full` vs `model(low,None)` → **identical** tensors / PSNR.

## Artifacts
- `/root/data/experiments/ASRT/eval_v4_fix_audit/stage_a_paired_dev34.json`
- `/root/data/experiments/ASRT/eval_v4_fix_audit/stage_b_v2_ref_dev30.json`

## Comparison rule
When comparing Ref gains, use the **same n=30** Correct Nano subset only. Do not average n=34 paired with n=30 ref tables as one number.
