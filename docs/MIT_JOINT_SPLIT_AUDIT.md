# MIT Joint Split Audit (testing-as-train)

Generated: 2026-10-30T05:26:16.644612+00:00

## Source

- Pool: MIT **testing** only (`/root/data/datasets/MIT/testing`)
- Low: `INPUT_IMAGES/{id}_N1.5`
- GT: `expert_c_testing_set/{id}`
- Ref: `nanobanana_ref_mit_v1/{id}_N1.5`
- Dropped missing Nano: **4** (examples ['a0226-WP_CRW_4861', 'a2516-jmac_DSC3843', 'a4968-Duggan_080819_1132', 'a5000-kme_0204'])
- Dropped missing Low: **0**

## Splits

| Split | n |
|-------|---|
| Ref Dev | 50 |
| Ref Train | 1127 |
| NoRef Train | 1127 |
| Triplet pool | 1177 |

## Intersection checks

- **Dev∩RefTrain**: PASS (n=0)
- **Dev∩NoRefTrain**: PASS (n=0)

## Notes

- **No official MIT test eval** — testing images are the training pool.
- Manifest root: `/root/data/experiments/ASRT/manifests_mit_joint`
- Dev IDs sha256: `cf5716fbdf3954df`

