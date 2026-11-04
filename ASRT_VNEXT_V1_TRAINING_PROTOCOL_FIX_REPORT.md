# ASRT vNext V1 Training Protocol Fix Report

**Branch:** `vnext-v1-irp`  
**Base commit:** `544b12ca3d4f825745b1f5e32ef466550883bfed`  
**Protocol:** `irp_v1_train_v2`  
**Date:** 2026-11-04  

**Environment:** Python 3.10.20 · PyTorch 1.12.1+cu113 · CUDA 11.3 · GPU RTX 4090  

**Scope:** Training system / tests only — **no** IRP-MSA / prior / loss redesign.  
**Formal 80k training:** **NOT RUN**.

---

## 1. Problems and root causes

| Issue | Root cause |
|-------|------------|
| `step` incremented per micro-batch while scheduler/max_steps meant updates | Mixed micro vs optimizer semantics |
| First update could miss warmup LR | Scheduler advanced *after* step; init did not apply update-0 LR clearly for the train loop |
| Resume rebuilt DataLoader without batch cursor | Only RNG/generator saved; epoch-internal position lost |
| Partial accum grads not in checkpoint | Mid-accum interrupt not recoverable |
| Accum equivalence test compared identical paths | Did not concat micro-batches into one large batch |
| Halo crop aligned **outer** top to window grid | With halo=22, valid region (`top+halo`) misaligned to full-image windows |
| `IRPTrainer.load_checkpoint` weak vs factory IRP checks | Incomplete protocol compare |

---

## 2. Step semantics (v2)

```text
global_step / max_steps / step(alias)
    = completed Optimizer Updates

1 Training Update
    = grad_accum × (forward + backward)
    + 1 × optimizer.step   (if grads finite / not AMP-skipped)

micro_step     = completed micro-batch backwards
epoch          = sampler epoch
batch_cursor   = consumed batches in current epoch
skipped_update_count = AMP/non-finite skips (do NOT advance global_step or scheduler)
```

`max_steps=80000` with `grad_accum=4` ⇒ **80000** parameter updates.

Logging / val / ckpt periods use **`global_step`**.

---

## 3. Warmup + Cosine

For update index \(k = 0,1,\ldots,T-1\) (applied **before** the update):

\[
\mathrm{lr}(k)=
\begin{cases}
\mathrm{base}\cdot\dfrac{k+1}{W} & 0 \le k < W \\[6pt]
\eta_{\min}+(\mathrm{base}-\eta_{\min})\cdot\tfrac12\bigl(1+\cos(\pi\cdot p)\bigr)
& W \le k < T
\end{cases}
\]

\[
p=\min\Bigl(1,\dfrac{k-W}{\max(1,T-W)}\Bigr)
\]

Example (`base=1e-4`, `W=1000`, `T=80000`, \(\eta_{\min}=1e-6\)):

| Update \(k\) | LR |
|-------------|-----|
| 0 | \(1\times10^{-7}\) |
| 999 | \(1\times10^{-4}\) |
| 1000 | \(1\times10^{-4}\) (cosine start) |
| \(T-1\) | near \(\eta_{\min}\) |

Grad accum does **not** stretch the schedule (update-indexed).

### AMP skipped updates

After `unscale_`, if any grad is non-finite: skip `optimizer.step`, do **not** increment `global_step`, do **not** `mark_update_done`, clear grads, still `scaler.update()`, increment `skipped_update_count`.

---

## 4. Resumable sampler

`asrt/training/resumable_sampler.py` · `ResumableBatchSampler`

State:

```text
epoch, permutation, batch_cursor, seed, batch_size, drop_last, dataset_size
```

- Trainer advances `batch_cursor` only after a batch is **consumed**.
- Prefetch (`num_workers>0`) does not mutate consumed cursor until the batch is used.
- Resume restores permutation + cursor; rebuilds DataLoader; does **not** re-roll RNG to invent a new perm.
- Dataset `set_epoch(sampler.epoch)` on epoch changes; `persistent_workers=False`.

---

## 5. Checkpoint / resume flow

1. Train only saves on **update boundaries** (after a full accum group).
2. Checkpoint includes: model/opt/sched/scaler, `global_step`/`micro_step`, `sampler_state`, RNG, `irp_hparams`, `train_protocol`, manifest hashes, `protocol_version`, git commit/dirty.
3. Resume: compare IRP hparams (`compare_irp_hparams`) + train protocol (`compare_train_protocol`); restore sampler; apply `lr_for_update(global_step)`.
4. Config mismatch → hard error (no silent override). Weights-only load is a separate explicit path, not “resume”.

---

## 6. Gradient accumulation equivalence

Path A: \(\sum_i \nabla L_i / N\) · Path B: \(\nabla L(\mathrm{cat\ batch})\)

| Metric | Result | Status |
|--------|--------|--------|
| Max grad diff | ~9e-8 | **PASS** |
| Max weight diff after Adam step | ~4e-6 | **PASS** |

AMP off, no dropout.

---

## 7. Interrupted vs uninterrupted

| Setting | Sample/aug IDs | Max param diff | Status |
|---------|----------------|----------------|--------|
| CPU, AMP off, accum=1, workers=0 | exact match | **0.0** | **PASS** |
| CPU, AMP off, accum=2, workers=0 | exact match | **0.0** | **PASS** |
| CUDA, AMP on, workers=2 | exact match | ~1.4e-5 | **PASS** (soft float) |

---

## 8. A2 / A3 consistency

| Check | Status |
|-------|--------|
| Param count equal | **PASS** |
| Same seed → shared init weights on common keys | **PASS** |
| Same sample_id sequence at step k | **PASS** |
| Same aug metadata | **PASS** |
| Separate `out_dir` | **PASS** |
| Only `prior_source` differs (low vs nano) | by construction **PASS** |

---

## 9. Halo crop window alignment

Valid region origin satisfies:

```text
(valid_top % window_size) == 0
(valid_left % window_size) == 0
outer = valid - halo
```

With halo=22, window=8: outer is **not** grid-aligned; valid **is**. Tests: **PASS**.

**Limit:** After flip/rotation, the valid crop’s window lattice is relative to the **augmented** tensor, not the original full-image orientation. Full-image inference without the same flip/rot cannot share that lattice. Documented limitation — not claimed bit-equal to arbitrary full-image window origins under geometric aug.

---

## 10. Checkpoint config validation

Resume checks model family keys + train protocol (optimizer, LR, WD, betas, micro/effective batch, accum, crop, halo, loss, warmup/max_steps/eta_min, AMP, seed, manifest hashes, dataset size, prior_source). Mismatch → **FAIL** with listed conflicts. Unit test **PASS**.

---

## 11. Regression

All prior IRP unit suites (attention/prior/shapes/grad/conditioning/source/inference/modes) **PASS**.  
Legacy Stage-A path not re-broken by trainer-only changes (**NOT RUN** full Stage-A train suite).

---

## 12. Remaining reproducibility limits

| Item | Status |
|------|--------|
| `num_workers>0` bit-exact weights | Soft (~1e-5 under AMP); sample/aug sequence **PASS** |
| Mid-accum kill recovery | **Not supported** — last boundary ckpt only |
| cudnn nondeterminism | May affect soft float diffs |
| Formal 80k + val selection | **NOT RUN** |

---

## 13. Test matrix

| Test | Status |
|------|--------|
| `test_irp_scheduler_steps` | **PASS** |
| `test_irp_sampler_resume` | **PASS** |
| `test_irp_accum_equivalence` | **PASS** |
| `test_irp_exact_resume` | **PASS** |
| `test_irp_config_consistency` | **PASS** |
| `test_irp_a2_a3_sequence` | **PASS** |
| `test_irp_halo_crop` (+ valid align) | **PASS** |
| `test_irp_training_resume` | **PASS** |
| `test_irp_checkpoint_protocol` | **PASS** |
| `test_irp_data_reproducibility` | **PASS** |
| Full IRP regression pack | **PASS** |
| 80k formal train | **NOT RUN** |

---

## 14. Git

**Hash:** `34c9b69f135838869aeb67f062180fc9e6643af9`

### Key files

```text
asrt/training/irp_trainer.py
asrt/training/resumable_sampler.py
asrt/data/halo_crop.py
asrt/data/irp_dataset.py
configs/asrt_irp_v1.yaml
tests/test_irp_exact_resume.py
tests/test_irp_scheduler_steps.py
tests/test_irp_sampler_resume.py
tests/test_irp_accum_equivalence.py
tests/test_irp_config_consistency.py
tests/test_irp_a2_a3_sequence.py
ASRT_VNEXT_V1_TRAINING_PROTOCOL_FIX_REPORT.md
```

---

## 15. Formal training entry (after freezing max_steps)

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ttsr
cd /root/projects/ASRT
PYTHONPATH=. python scripts/train_irp.py \
  --config configs/asrt_irp_v1.yaml \
  --max_steps 80000

# Resume (same config / protocol):
PYTHONPATH=. python scripts/train_irp.py \
  --config configs/asrt_irp_v1.yaml \
  --resume /path/to/checkpoints/last.pt \
  --max_steps 80000
```

A2/A3: duplicate config with `model.prior_source: low|nano` and **distinct** `out_dir`.

---

## Conclusion

**YES — training protocol is ready for formal A1/A2/A3 runs** under:

- `protocol_version: irp_v1_train_v2`
- `max_steps` = optimizer updates
- `num_workers=0` for strongest resume determinism (default in config)
- Full-image eval
- Distinct out_dirs per prior_source

Still required before claiming research results: freeze LR/warmup/steps, run 80k, select `best_validation_psnr` on the fixed val manifest — **not** done in this task.
