# ASRT-v4 Reliability Diagnostics

## Gates

- `C_texture`, `C_illum` ∈ [0,1], predicted by **independent** heads on a shared encoder.
- Multiscale: bilinear upsample to H/2 and H (`C_T`); H/4 & H/2 for `C_L` (IGAB scales only).
- Residual form: `F + C·Δ`, `L + C·Δ` so **C=0 ⇒ exact zero reference residual** (zero-init last adapter layer at start).

## Logged statistics (train.jsonl)

When reference is used, each log row may include:

```json
"c_t": {"mean", "std", "p10", "p50", "p90", "frac_near0", "frac_near1"},
"c_l": {…},
"match": {"entropy", "max_weight", "disp_mag", "disp_var", "continuity", "q_norm", "k_norm"},
"tex_res_norm": {"coarse", "middle", "fine"}
```

### Degeneration checks
- `frac_near0 → 1` and `tex_res_norm → 0`: model ignoring reference (safe but no gain).
- `frac_near1 → 1` with high texture residual in dark flats: possible mottling risk (compare to B-v2 failure modes).
- `C_T` vs `C_L` spatial correlation: should **not** be identically 1 (heads are separate).

## Matching diagnostics

| Signal | Meaning |
|---|---|
| Matching entropy (normalized) | Soft ambiguity; high → uncertain warp |
| Max weight | Peakiness of local softmax |
| Expected displacement magnitude | Geometry shift in H/4 cells |
| Displacement variance | Soft spread around expected disp |
| Continuity | Neighboring-cell disp jump (spatial smoothness) |
| Q/K norms | After L2-normalize ≈1; pre-norm via logged maps |

## Visualization recipe (post-train)

For each Dev case export: Low, Nano Ref, GT, Enhanced(ref), Enhanced(no-ref), disp field, entropy, C_T / C_L heatmaps, texture residual norm map, aligned P_ref.

Priority scenes: dark walls/floors, grass, text/edges, geometry-changed ref regions.

## Pre-train sanity (CUDA smoke)

With random weights / Stage A init + zero adapters: ref≈no-ref until reliability opens and adapters leave zero. After 3 Adam steps loss decreased (0.944→0.936) with finite grads — gates not yet evaluated on Dev (requires training).
