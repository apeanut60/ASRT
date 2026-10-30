# Training

## Stages

### Stage A — restore without reference

- Data: LOLv2 Real + Syn Low/GT; `ref=None` (`no_ref_prob: 1.0`).
- Optimizer: Adam `2e-4`, L1, Cosine → `1e-6`, planned 80k steps.
- Crops: Real 256 / Syn 128 (config fields `crop` / `crop_syn`).
- Config: `configs/stage_a_lolv2_restore.yaml`.

### Stage B — add Nano reference

- Data: Real∩Nano + Syn∩Nano triplets; ~20–30% no-ref batches (`no_ref_prob≈0.25`).
- Grouped LR: backbone low, inject heads higher (`lr_backbone` / `lr_inject`).
- Warm-start from Stage A best. Planned 100k.
- Config: `configs/stage_b_lolv2_reference.yaml`.

### Stage C — domain FT

- Huawei / Nikon ~500 Nano triplets each; 20k–40k + early stop from B ckpt.
- Do **not** pull full LSRW Low/GT into the main loop.
- Configs: `configs/stage_c_huawei_ft.yaml`, `configs/stage_c_nikon_ft.yaml`.

## Trainer features

- AMP + GradScaler (CUDA), grad clip start `1.0` with clip-event logging.
- Gradient accumulation: `accum` micro-batches per optimizer step (`loss` scaled by `1/accum`). Stage A default: `batch_size=8`, `accum=1`. `max_steps` counts optimizer steps.
- Checkpoints: model / optim / sched / scaler / RNG / cfg / step.
- Resume: `python scripts/train.py --config ... --resume path/to/last.pt`.

## Smoke

```bash
python scripts/train.py --config configs/stage_a_lolv2_restore.yaml \
  --max_steps 30 --out_dir /root/data/experiments/ASRT/smoke_a
```

Full 80k/100k is not the default deliverable of the scaffold round.
