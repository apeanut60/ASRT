# ASRT-v4 Accumulation Tests

Suite: `tests/test_gradient_accumulation.py` (+ protocol suite).

| Test | What it proves | Result |
|---|---|---|
| **A** Trainer.train batch=2 accum=2 | 2 distinct micros, 4 dual forwards, **1** opt step, step=1, samples=4 | PASS |
| **B** grad equivalence | batch2×accum2 grads/weights == batch4×accum1 (atol 1e-5) | PASS |
| **B-detect** | old “last micro only ÷accum” grads **≠** correct — regression detector | PASS |
| **C** mixed ref/noref micros | both micros add nonzero grad to shared weights | PASS |
| **D** AMP skip | scale↓ ⇒ scheduler not stepped | PASS |
| match_dim≠n_feats | ValueError | PASS |
| absolute eta_min | all groups → 1e-6 at T | PASS |

Protocol suite `tests/test_trainfix_protocol.py`: seed, eval auto/full/tiled, hparam reject, Trainer.evaluate≡protocol PSNR, LR preview.

Smoke real data (`stage_b_v4_trainfix_smoke`, 40 steps):

```
micros=2 samples=8 effective_batch=8
optimizer_steps=39 amp_skips=1
eval n=30 PSNR≈21.84 (auto protocol)
```

**Old bug would show:** micros fetched but only 1 forward-path micro contributing; samples_processed logged as 4 with loss/2 scaling — Test B-detect catches that pattern.
