# Syn Joint Split Audit

Generated: 2026-10-28T11:35:12.291001+00:00

## Canonical Dev

- Source: Nano-capable Syn Train sample (seed=42, n=30)
- Ref Dev: **n=30**
- NoRef Dev (same IDs): **n=30**

## Train sizes

| Split | n |
|-------|---|
| Syn Ref Train | 220 |
| Syn NoRef Train | 870 |
| Official Syn Test | 100 |

## Intersection checks (must be empty)

- **Dev∩RefTrain**: PASS (n=0)
- **Dev∩NoRefTrain**: PASS (n=0)
- **Dev∩OfficialSynTest**: PASS (n=0)
- **RefTrain∩OfficialSynTest**: PASS (n=0)
- **NoRefTrain∩OfficialSynTest**: PASS (n=0)

## Notes

- Old manifests under `manifests/` / `manifests_b_v2/` are **not** overwritten.
- New root: `/root/data/experiments/ASRT/manifests_syn_joint`
- Ref Dev IDs sha256: `f08d04bb5fea825f`

