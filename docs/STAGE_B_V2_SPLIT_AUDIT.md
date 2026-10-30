# Stage B-v2 Split Audit

Generated: 2026-10-21T12:18:58.939193+00:00

## Canonical Dev

- Source: Stage A `lolv2_real_paired_dev` (**n=34**)
- Stage A Dev ∩ Stage A Train: **0** (must be 0)
- Ref-capable Dev (has Nano): **n=30** (IDs without Nano excluded from ref eval, never GT-filled)
- Missing Nano among Dev: ['low00282', 'low00319', 'low00328', 'low00500']

## Train sizes (after excluding ALL Dev IDs + Test IDs)

| Split | n |
|-------|---|
| Real Ref Train | 609 |
| Syn Ref Train | 250 |
| Real NoRef Train | 655 |
| Syn NoRef Train | 855 |

## Intersection checks (must be empty)

- **Dev∩RealRefTrain**: PASS (n=0) examples=[]
- **Dev∩SynRefTrain**: PASS (n=0) examples=[]
- **Dev∩RealNoRefTrain**: PASS (n=0) examples=[]
- **Dev∩SynNoRefTrain**: PASS (n=0) examples=[]
- **Dev∩OfficialTest**: PASS (n=0) examples=[]
- **RealRefTrain∩OfficialTest**: PASS (n=0) examples=[]
- **RealNoRefTrain∩OfficialTest**: PASS (n=0) examples=[]
- **SynRefTrain∩OfficialTest**: PASS (n=0) examples=[]
- **OldTriDev∩NewRealNoRefTrain(should_be_0_now)**: FAIL (n=30) examples=['low00011', 'low00032', 'low00056', 'low00066', 'low00070', 'low00087', 'low00091', 'low00112', 'low00136', 'low00207']
- **A_Dev∩A_Train(must_0)**: PASS (n=0) examples=[]

## Notes

- Old Stage B `triplet_dev ∩ real_paired_train` leakage is **eliminated** in this version.
- Old manifests under `/root/data/experiments/ASRT/manifests/` are **not overwritten**.
- New root: `/root/data/experiments/ASRT/manifests_b_v2`

