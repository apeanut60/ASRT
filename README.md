# ASRT

Illumination-Guided Cross-Scale Reference Restoration for low-light enhancement.

Standalone package: CSFI-UNet backbone + optional Nano reference via LTE/SearchTransfer.
`forward(low, ref=None)` fully bypasses reference compute when `ref` is omitted.

## Install

```bash
# Prefer the existing CUDA torch env on this machine:
conda activate ttsr
cd /root/projects/ASRT
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# 1) Audit data & write manifests
python scripts/audit_data.py --out /root/data/experiments/ASRT/manifests

# 2) Unit tests
python -m unittest discover -s tests -v

# 3) Stage A smoke (no reference, small steps)
python scripts/train.py \
  --config configs/stage_a_lolv2_restore.yaml \
  --max_steps 30 \
  --out_dir /root/data/experiments/ASRT/smoke_a

# 4) Evaluate (bypass / correct Nano)
python scripts/evaluate.py \
  --config configs/stage_a_lolv2_restore.yaml \
  --ckpt /root/data/experiments/ASRT/smoke_a/checkpoints/last.pt \
  --manifest /root/data/experiments/ASRT/manifests/lolv2_real_paired_dev.json \
  --mode bypass --limit 8
```

## Stages

| Stage | Config | Role |
|-------|--------|------|
| A | `configs/stage_a_lolv2_restore.yaml` | LOLv2 Real+Syn Low/GT, `ref=None`, ~80k |
| B | `configs/stage_b_lolv2_reference.yaml` | Real∩Nano + Syn∩Nano, ~25% no-ref, ~100k |
| C | `configs/stage_c_{huawei,nikon}_ft.yaml` | Domain FT from B, ~500 Nano triplets |

This repo ships scripts and YAML for full runs. **Long 80k/100k training and official PSNR numbers are not claimed here** unless you launch them.

## Layout

```text
asrt/models/     ASRT, CSFIUNet, IGAB, LTE/SearchTransfer
asrt/data/       paired / triplet / manifest (never GT-as-Ref)
asrt/training/   AMP trainer, grouped LR, resume
asrt/evaluation/ RGB PSNR/SSIM, full/tiled inference
configs/         Stage A/B/C YAML
scripts/         audit_data, train, evaluate, convert_legacy_weights
docs/            architecture, training, evaluation, provenance
```

## Known limits (this delivery)

- Full Stage A/B schedules are configured but not run to completion in smoke.
- Official Test PSNR/SSIM not reported until a full train completes.
- Stage C expects audited Huawei/Nikon Nano manifests; incomplete Nano → those rows stay out of the reference branch (no GT substitute).

See `docs/` for details.
