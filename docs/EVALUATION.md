# Evaluation

## Protocol

- Metrics: RGB **PSNR** / **SSIM** on `[0,1]` (uint8 path), `crop_border=0`, mean over images.
- **Forbidden**: GT mean-align, GT-as-Ref, reporting only luminance PSNR as primary.
- Dev split: fixed fraction from Train (`split_dev`, seed 42). Official Test only for final reports.

## Modes (`scripts/evaluate.py`)

| `--mode` | Behavior |
|----------|----------|
| `bypass` | `ref=None` |
| `correct` | paired Nano when present |
| `both` | report both |
| `diag` | Correct / Bypass / Mismatch deltas |

Inference: `--tiled` for large images; global illum (if enabled) runs once on the full frame after tiling.

## Example

```bash
python scripts/evaluate.py \
  --config configs/stage_b_lolv2_reference.yaml \
  --ckpt /path/to/best.pt \
  --manifest /root/data/experiments/ASRT/manifests/lolv2_real_triplet_dev.json \
  --mode both --limit 32
```
