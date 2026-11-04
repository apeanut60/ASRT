# ASRT vNext V1 — Formal A2/A3 Training Report

**Status:** `IN PROGRESS` (A2 training running; A3 queued)  
**Code freeze (configs/guards):** `7267cf9097a3839dd00b2d10c20d9e116e1d455e`  
**Helpers commit (launch/eval/plot):** `4a72f6a` (after A2 start; does not alter model/trainer math)  
**Review baseline:** `95c17509eb5895705045c875fdeea77c26caf004`  
**Protocol:** `irp_v1_train_v3`  
**Updated:** 2026-11-04

> Final PSNR / paired deltas / diagnostics / plots are **NOT RUN** until both 80k runs finish.  
> Do not treat any partial numbers below as Nano-Prior effectiveness evidence.

---

## 1. Git Commit

| Item | Value |
|------|-------|
| Formal config freeze | `7267cf9` |
| A2 `resolved_config.json` git_commit | `7267cf9097a3839dd00b2d10c20d9e116e1d455e` |
| A2 recorded `git_dirty` at launch | `true` (uncommitted helper scripts only; model/trainer already frozen) |
| Branch | `vnext-v1-irp` |

---

## 2. Complete training configuration

Configs (differ **only** by `out_dir` + `model.prior_source`):

- `configs/asrt_irp_v1_a2.yaml` → `prior_source: low`
- `configs/asrt_irp_v1_a3.yaml` → `prior_source: nano`

| Parameter | Value |
|-----------|-------|
| Init | Scratch, seed 42 |
| Optimizer | Adam, lr 1e-4, wd 1e-4, betas (0.9, 0.999) |
| Scheduler | Warmup 1000 + Cosine, eta_min 1e-6 |
| Updates | 80000 |
| Loss | MSE on valid 128×128 (input patch 192×192 with halo 32) |
| Batch | **8 × accum 1** (effective 8) — confirmed by preflight |
| AMP / grad clip | true / 1.0 |
| ref_mode | correct |
| Validation | full-image, every 2000 updates |
| Checkpoint | every 2000 + best_validation_psnr + last |

---

## 3. Dataset / Reference Manifest

| Manifest | Path |
|----------|------|
| Train | `lolv2_real_triplet_train.json` (607) |
| Val | `lolv2_real_triplet_dev.json` (32) |
| Nano Ref | `/root/data/datasets/lol-v2-real/Train/nanobanana_ref_v2` |

Preflight + A2 `data_fingerprint.json` written under experiment dirs.

---

## 4. Train / Validation counts

| Split | N | Notes |
|-------|---|-------|
| Train | 607 | All have Nano Ref files; train∩val = ∅ |
| Val | 32 | All 600×400; refs present |

**Dev-set caveat:** `lolv2_real_triplet_dev.json` was used during engineering smokes. It is **not** an independent paper test set.

---

## 5. A2/A3 initial weight equality

Preflight (`scripts/preflight_irp_formal.py`):

```text
params=3847371, keys=297, equal=True
prior_a2=low, prior_a3=nano
```

Independent scratch init (no A2→A3 weight load).

---

## 6. GPU Preflight

| Check | Result |
|-------|--------|
| Status | **PASS** |
| Batch plan | batch=8, accum=1 (no OOM) |
| Peak memory | ~6646 MiB (train probe) |
| A2/A3 20 updates | finite loss/grads; ref-branch grads nonzero |
| Full-image val (n=2) | OK |
| Fingerprint match A2/A3 | True |
| Artifacts | `/root/data/experiments/ASRT/vnext_v1/preflight/` |

Preflight checkpoints are **not** used to initialize formal runs.

---

## 7. Effective Batch & memory (formal)

| Item | Value |
|------|-------|
| Effective batch | 8 |
| Formal A2 GPU mem (early) | ~8894 MiB |
| Device | RTX 4090 24GB |

---

## 8. Optimizer Update progress

| Run | out_dir | Status |
|-----|---------|--------|
| A2 | `.../A2_low_prior_seed42` | **IN PROGRESS** (tmux `asrt_formal`) |
| A3 | `.../A3_nano_prior_seed42` | **NOT RUN** (queued after A2) |

Observed early A2 log: step 1 → 50; ~2.4 s/update ⇒ rough ETA ~50–60 h per 80k run (plus val).

Launch:

```bash
tmux attach -t asrt_formal
# or logs:
tail -f /root/data/experiments/ASRT/vnext_v1/logs/a2_train_stdout.log
```

Orchestrator: `scripts/run_formal_a2_a3.sh`

---

## 9–16. Curves / Best ckpt / Paired ΔPSNR / Bootstrap / Diagnostics / Viz / Limits

| Section | Status |
|---------|--------|
| 9. Loss curves | **IN PROGRESS** (partial `train_log.jsonl`) |
| 10. Val PSNR/SSIM | **NOT RUN** until first val @ 2000 |
| 11. Best/Last ckpt | **NOT RUN** |
| 12. Paired ΔPSNR | **NOT RUN** — `scripts/eval_a2_a3_compare.py` ready |
| 13. Bootstrap 95% CI | **NOT RUN** |
| 14. A3 reference diagnostics | **NOT RUN** |
| 15. Visualizations | **NOT RUN** — `scripts/plot_a2_a3_curves.py` ready |
| 16. Failures/limits | See below |

### Known limits (current)

- Single seed (42); validation is exploratory, not paper test.
- A2 launched with `git_dirty=true` due to uncommitted helpers; training code hash is `7267cf9`.
- Full 80k×2 wall time is multi-day on one 4090; do not invent final metrics.

---

## 17. Follow-ups (after both runs complete)

1. Run `eval_a2_a3_compare.py --diagnostics` on `best_validation_psnr.pt`.
2. Run `plot_a2_a3_curves.py` + qualitative grids.
3. Fill sections 9–15 from on-disk logs/ckpts only.
4. Multi-seed / held-out test confirmation before claiming Nano Prior gains.

---

## Engineering delivered this task

- Frozen A2/A3 YAML configs
- Out-dir overwrite guard + recursive Adam resume test
- GPU preflight PASS @ batch 8
- Formal A2 started under tmux; A3 auto-starts after A2
- Compare/plot scripts committed

**READY for continued unattended training:** YES  
**READY to claim Nano > Low:** NO (training incomplete)
