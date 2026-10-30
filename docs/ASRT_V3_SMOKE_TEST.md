# ASRT-v3 Smoke Test

Date: 2026-10-22

## Unit tests

```bash
cd /root/projects/ASRT
PYTHONPATH=. python tests/test_asrt_v3.py
```

Result: **ALL_PASS** (11 tests) — ref bypass, zero-init Adapter/CondRB, C_T=0 / C_L=0 gates, match warp, grad, B-v2 load, optimizer coverage, FP16.

## Integration smoke (4 steps)

```bash
python -u scripts/train.py \
  --config configs/stage_b_v3_full.yaml \
  --out_dir /root/data/experiments/ASRT/stage_b_v3_full_smoke \
  --max_steps 4 \
  --pretrained /root/data/experiments/ASRT/stage_b_v2/checkpoints/best.pt
```

Observed:

- `v3 warm-start from B-v2: loaded=242 missing_new=28`
- Teacher Stage A loaded (`w=0.1`)
- `resume_optim=False`, step resets to 0
- step1 loss≈0.075, mem≈7.0GB, batch=8
- eval n=30 Dev PSNR≈21.26（稳定期初值，接近 B-v2 best）

## Full training command

```bash
cd /root/projects/ASRT
nohup python -u scripts/train.py \
  --config configs/stage_b_v3_full.yaml \
  --out_dir /root/data/experiments/ASRT/stage_b_v3_full \
  --pretrained /root/data/experiments/ASRT/stage_b_v2/checkpoints/best.pt \
  > /root/data/experiments/ASRT/stage_b_v3_full/train.stdout.log 2>&1 &
```

Do **not** overwrite Stage A / B-v1 / B-v2 experiment dirs.
