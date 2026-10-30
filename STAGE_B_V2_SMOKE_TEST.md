# Stage B-v2 Smoke Test Report

Date: 2026-10-21

## Checks

| # | Check | Result |
|---|-------|--------|
| 1 | Unit tests `test_query_illum_align` (+ bypass/tlv) | **15/15 OK** |
| 2 | Manifest intersection audit | **ALL CHECKS PASSED** (see SPLIT_AUDIT) |
| 3 | Stage A `best.pt` step | **58000**, Dev PSNR 21.427 |
| 4 | Strict `load_state_dict` into B-v2 model | **OK** (all keys) |
| 5 | Query align cfg enabled | `gamma_hist`, α=0.75, β=0.50 |
| 6 | Ref fwd/bwd + noref fwd | **OK** |
| 7 | GPU bs=8 crop256 + align peak | ~**7.1 GB** (4090 OK) |
| 8 | 4-step train smoke | **OK**; `batch=8`; eval n=**30** (ref Dev) |

## Smoke command

```bash
conda activate ttsr
cd /root/projects/ASRT
python -u scripts/train.py \
  --config configs/stage_b_v2_lolv2_reference.yaml \
  --max_steps 4 \
  --out_dir /tmp/asrt_b_v2_smoke \
  --pretrained /root/data/experiments/ASRT/stage_a/checkpoints/best.pt
```

## Full training command

```bash
conda activate ttsr
cd /root/projects/ASRT
mkdir -p /root/data/experiments/ASRT/stage_b_v2
nohup python -u scripts/train.py \
  --config configs/stage_b_v2_lolv2_reference.yaml \
  --out_dir /root/data/experiments/ASRT/stage_b_v2 \
  --pretrained /root/data/experiments/ASRT/stage_a/checkpoints/best.pt \
  > /root/data/experiments/ASRT/stage_b_v2/train.stdout.log 2>&1 &
```

Output path: `/root/data/experiments/ASRT/stage_b_v2/`

## Notes

- Optimizer starts fresh (`resume_optim=False`); step resets to 0.
- Does **not** resume old Stage B 56k/74k.
- Ablation (align off, same split): `configs/stage_b_v2_baseline_no_align.yaml`.
