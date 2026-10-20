# Smoke results (scaffold delivery)

Environment: `conda` env `ttsr`, CUDA RTX 4090, torch 1.12.1+cu113.

## Unit tests

```text
python -m unittest discover -s tests -v
# 15 tests OK (odd size, no-ref, native T_lv*, resume, metrics, geometry, AMP)
```

## Data audit

| Split | n | missing_ref | notes |
|-------|---|-------------|-------|
| lolv2_real_paired | 689 | 0 | |
| lolv2_real_triplet | 639 | 50 | incomplete Nano dropped |
| lolv2_syn_paired | 900 | 0 | |
| lolv2_syn_triplet | 250 | 650 | incomplete Nano dropped |
| lolv2_real_test_* | 100 | 0 | |
| huawei_triplet | 500 | 1950 | Nano budget only |
| nikon_triplet | 500 | 2650 | Nano budget only |

Manifests: `/root/data/experiments/ASRT/manifests/` (+ `audit_summary.json`).

## GPU smoke

| Run | steps | peak mem | notes |
|-----|-------|----------|-------|
| Stage A | 30 + resume→40 | ~707 MB | L1 falling; eval PSNR ~14 (smoke only) |
| Stage B | 20 from A ckpt | ~724 MB | ref=True batches; eval both modes |
| Stage C Huawei | 10 | ~1656 MB | tiled eval for large frames; OK |
| Stage C Nikon | 5 | — | see `smoke_c_nikon/summary.json` |

**Not done:** full 80k / 100k Stage A/B, official Test PSNR claims.

**Remote:** `gh` CLI not installed on this host — no `apeanut60/ASRT` remote created. Local `.git` is initialized.
