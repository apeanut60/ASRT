# ASRT-v3 Training Report

## Setup

| Item | Value |
|------|-------|
| Init | Stage B-v2 `best.pt` @ **70k**（严格旧键；28 个新键缺失） |
| Optim | **全新** Adam 分组 + Cosine；不继承 B-v2 优化器 |
| Teacher | Stage A 58k，`λ=0.1` consistency on no-ref |
| Stable | 前 **2000** step 仅训 `v3_new`（其余 LR=0） |
| Data | `manifests_b_v2`（干净划份） |
| Steps | 100000，bs=8，AMP FP16，seed=42 |
| Out | `/root/data/experiments/ASRT/stage_b_v3_full/` |

## LR groups

| Group | LR |
|-------|-----|
| `v3_new`（Reliability / Illum Adapter / FineCT / CondRB.adapter） | 1e-4 |
| backbone | 1e-5 |
| inject | 1e-4 |
| igab | 2e-5 |
| illum_tail | 2e-5 |

## Selection protocol

- 选模：干净 Ref Dev（n=30），RGB PSNR/SSIM，无 GT mean-align
- **不用**官方 Test 调参 / 选 checkpoint
- 对比应包含：A bypass / B-v2 bypass / B-v2 correct / v3 bypass / v3 correct

## Status

- Smoke：PASS（见 `ASRT_V3_SMOKE_TEST.md`）
- Full train：见 `stage_b_v3_full/train.stdout.log`；完成后回填 best step / Dev / 增益分布

## Reproduce

```bash
cd /root/projects/ASRT
python -u scripts/train.py \
  --config configs/stage_b_v3_full.yaml \
  --out_dir /root/data/experiments/ASRT/stage_b_v3_full \
  --pretrained /root/data/experiments/ASRT/stage_b_v2/checkpoints/best.pt
```
