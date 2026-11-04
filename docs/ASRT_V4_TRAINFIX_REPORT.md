# ASRT-v4 Training Fix Report

Baseline: `4f19228`. New commit covers Trainer accumulation, seed/RNG, eval protocol, hparam checks, absolute `eta_min`.

## P0 — Gradient Accumulation

**Root cause:** micro-batch `for` only fetched data; Forward/Backward ran **once** after the loop → only the last micro trained, while `loss/accum` still divided by 2.

**Fix:** `asrt/training/trainer.py` — full sample→forward→`loss/accum`→backward **inside** the micro loop; unscale/clip/opt/scaler/sched **once** outside.

| Config | Before (bug) | After |
|---|---|---|
| batch=4, accum=2 | 4 samples / opt step; loss÷2 | **8 samples** / opt step; both micros backward |
| reported effective_batch | lied as 8 | true 8 |

Smoke `stage_b_v4_trainfix_smoke` @40: `micros=2 samples=8`, `optimizer_steps=39`, `amp_skips=1`.

## AMP ↔ Scheduler

Detect skip via `scaler.get_scale()` decrease; **do not** `sched.step()` when skipped. Logged as `amp_step_skipped`.

## Seed / Resume

`seed_everything` before model+loaders; DataLoader `generator` + `worker_init_fn`. Checkpoint stores python/numpy/torch/cuda RNG + generator state.

**Limit (documented in ckpt):** resume does **not** restore DataLoader iterator cursor → not bitwise-identical ordering after interrupt.

## Eval protocol

`asrt/evaluation/protocol.py` — `evaluation.inference_mode: auto|full|tiled`. Trainer.evaluate and `scripts/evaluate.py` share it. Ref mode skips missing-ref (no silent no-ref mix).

## Checkpoint hparams

v4 load compares non-weight keys (`match_radius`, `n_feats`, …). Conflict → refuse unless `--allow_hparam_override`.

## match_dim

`match_dim != n_feats` → `ValueError` (no new projections).

## Scheduler `eta_min`

Now **absolute floor per group**: all groups → `eta_min` at T. Preview at 0/50/100% logged at train start.

## Historical impact

| Experiment | accum in cfg | Affected by bug? |
|---|---|---|
| Stage A | 1 | **No** (micro loop with accum=1 was correct) |
| Stage B / B-v2 | 1 | **No** |
| ASRT-v3 / joint | 1 | **No** |
| ASRT-v4 (`stage_b_v4*`) | **2** | **Yes** — under-batched; discard for fair protocol |

Keep dirs; do not resume them as formal v4. New formal run: `configs/stage_b_v4_trainfix.yaml` → `stage_b_v4_trainfix/`.

## Answers

1. Accumulation **yes** (tests A/B + smoke samples=8).  
2. batch4×accum2 → **8 images** in gradient.  
3. Ref dual + pure no-ref both supported; stats per micro.  
4. AMP skip blocks scheduler.  
5. Shared eval protocol + strict/hparam load.  
6. Old A/B/v3 weights OK for inference; v4 prior trains protocol-wrong.  
7. Ready to restart from Stage A @58k via trainfix config.

```bash
python scripts/train.py --config configs/stage_b_v4_trainfix.yaml
```
