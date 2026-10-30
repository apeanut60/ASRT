# ASRT-v4 Reproducibility Audit

## Seeding
- `scripts/train.py` and `Trainer.__init__` call `seed_everything(cfg.seed)`.
- DataLoader: `torch.Generator` + `worker_init_fn=seed_worker`.
- Same seed → identical first `torch.rand` sequence (unit test).

## Checkpoint contents (trainfix)
- model / optimizer / scheduler / GradScaler
- step, optimizer_steps, amp_skips
- python / numpy / torch / cuda RNG
- DataLoader generator state
- cfg + eval_cfg
- explicit resume note about iterator cursor

## Resume guarantees
| Restored | Guaranteed identical? |
|---|---|
| Weights, optim, sched, scaler | Yes (if same code) |
| RNG + generator | Yes |
| Next DataLoader batch index | **No** — iterators rebuilt |

## Eval reproducibility
Trainer and `evaluate.py` use `resolve_eval_cfg` / `infer_with_protocol`. Best ckpt selection records `eval_cfg`. Same ckpt + same mode → same PSNR (protocol unit test).

## Scheduler
Absolute `eta_min` per group. Preview example (trainfix, T=40 smoke override in preview used max_steps): at T all groups → 1e-6.

## Config constraint
`match_dim` must equal `n_feats` (documented in yaml comments).

## Experiment isolation
| Dir | Status |
|---|---|
| stage_b_v4 | stopped ~800; accum bug; keep |
| stage_b_v4_fixed / _smoke | pre-accum-fix; keep |
| stage_b_v4_trainfix_smoke | post-fix smoke OK |
| stage_b_v4_trainfix | formal (not auto-started) |
