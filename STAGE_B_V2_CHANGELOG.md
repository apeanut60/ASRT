# Stage B-v2 Changelog

## What changed

### A. Query Illumination Alignment (matching only)

- Ported `query_illum_align.py` from `TTSR-lowlight-v1-data1SOTA` into `asrt/models/query_illum_align.py`.
- Wired into `ASRT.match_reference`: aligned Low → LTE Query; **Raw Nano Ref → K/V**; **Raw Low → backbone**.
- Config switch: top-level `query_alignment.enabled` (default off for old YAMLs; **on** in `stage_b_v2_lolv2_reference.yaml`).
- Full-image gamma/LUT estimated in `TripletRefDataset` **before** crop; overrides passed through training batch.
- Tiled inference estimates maps once on full image and reuses per tile (`asrt/evaluation/inference.py`).

**Parameters used:** `gamma_alpha=0.75`, `hist_beta=0.50`, `gamma_min=0.3`, `gamma_max=1.0`, `hist_bins=256`.  
Provenance: TTSR-lowlight-v1-data1SOTA v1r7 / v2_p2 working recipe. Task fallback `beta=0.20` was **not** used because historical effective `beta=0.50` is recoverable from that codebase.

### B. Train/Dev isolation

- New manifests under `/root/data/experiments/ASRT/manifests_b_v2/` (old manifests untouched).
- Canonical Dev = Stage A `lolv2_real_paired_dev` (n=34), verified disjoint from Stage A train.
- Ref Dev = Nano-capable subset (n=30); IDs without Nano excluded (never GT-filled).
- All B-v2 train loaders exclude **all** canonical Dev IDs.
- See `docs/STAGE_B_V2_SPLIT_AUDIT.md`.

### C. Training recipe

- Fresh Stage B-v2 from Stage A **58k best.pt** (strict load, new optim/sched).
- Same recipe as B-v1 otherwise: 100k, bs=8, L1, grouped LR, `no_ref_prob=0.25`, seed=42.
- Ablation YAML without align: `configs/stage_b_v2_baseline_no_align.yaml` (same clean split).

## What did NOT change

- CSFI-U-Net / IGAB / Gate formula / L1 loss / LTE freeze policy.
- No Reference-conditioned IGAB.
- Old Stage A / Stage B experiment directories not overwritten.

## Files touched

- `asrt/models/query_illum_align.py` (new)
- `asrt/models/asrt.py`
- `asrt/data/triplet.py`
- `asrt/training/trainer.py`
- `asrt/evaluation/inference.py`
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/build_stage_b_v2_manifests.py`
- `configs/stage_b_v2_*.yaml`
- `tests/test_query_illum_align.py`
