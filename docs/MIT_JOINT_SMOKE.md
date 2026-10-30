# MIT Joint v3 Smoke (testing-as-train)

- Manifests: `/root/data/experiments/ASRT/manifests_mit_joint/` — Ref Dev 50 / Train 1127；缺 Nano 4 景已丢弃
- Audit: `docs/MIT_JOINT_SPLIT_AUDIT.md`
- Smoke 4-step OK（~4GB crop192）；A→v3 loaded=242 missing=28
- Full train: `/root/data/experiments/ASRT/joint_mit_v3/` — **不跑官方 MIT test 分数**

```bash
python -u scripts/train.py \
  --config configs/joint_mit_v3.yaml \
  --out_dir /root/data/experiments/ASRT/joint_mit_v3 \
  --pretrained /root/data/experiments/ASRT/stage_a/checkpoints/best.pt
```
