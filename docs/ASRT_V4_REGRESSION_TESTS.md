# ASRT-v4 Regression Tests (post-fix)

## Suites run

| Suite | Result |
|---|---|
| `tests/test_asrt_v4_fixes.py` | **ALL PASS** |
| `tests/test_eval_protocol.py` | **OK** |
| `tests/test_reference_bypass.py` | **OK** |
| `tests/test_forward_backward.py` | **OK** |
| Selected `test_asrt_v3` (bypass, illum adapter) | **OK** |
| CUDA AMP smoke (in fixes suite) | **OK** |
| Stage A warm-start into fixed v4 | **OK** (230/89/12) |
| Fixed smoke train 60 steps | **OK** |

## Assertive fix checks (not smoke-only)

| Test | Assertion |
|---|---|
| Illum equal gray | `\|D_ref\| < 0.05` |
| Ref brighter / darker | `mean(D) > 0.5` / `< -0.5` |
| Gate=0 with **non-zero** adapter weights | residual exactly 0; gate=1 moves features |
| `ref=None` | RefEncoder/Matcher/Reliability raise if called |
| Factory | v4→legacy `RuntimeError`; v4 strict roundtrip |
| Metrics | endpoints + no auto range; identity PSNR inf |
| Tiled | `wt>0` everywhere; sizes 128×160 … 513×769; ref & no-ref |
| Pad geometry | output HW restored; T_h4 matches e2 |
| Dual loss | both branches produce grads on RefEnc/Q/Reli |
| Inject freeze | `requires_grad=False` on unused heads |

## Full / Tiled differences (MAE)

| Size | Ref | MAE | Max | Border MAE |
|---|---|---|---|---|
| 128×160 | Y/N | ~1.4e-8 | ~1.8e-7 | ~9e-9 |
| 200×300 | Y/N | ~2.2e-8 | ~1.8e-7 | ~1.6e-8 |
| 300×200 | Y/N | ~2.2e-8 | ~2.4e-7 | ~1.7e-8 |
| 255×257 | Y/N | ~3.1e-6 | ~1.3e-4 | ~1.4e-5 |
| 513×769 | Y/N | ~1.4e-6 | ~1.3e-4 | ~1.9e-5 |

Non-multiples of 4 show larger (still small) border discrepancy from pad/crop — expected; not claimed bit-exact.

## Sampling (smoke 60 steps)

```
ref_batches=45  pure_noref_batches=15
domain_real≈160  domain_syn≈80  (per-image domain tags in batches)
no_ref_prob=0.15 (configured; realized pure-noref share ≈25% on short run)
```
