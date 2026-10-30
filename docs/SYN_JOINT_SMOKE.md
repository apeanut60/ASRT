# Syn Joint v3 Smoke

- Manifests: `/root/data/experiments/ASRT/manifests_syn_joint/` (Ref Dev 30 / Ref Train 220 / NoRef Train 870)
- Audit: `docs/SYN_JOINT_SPLIT_AUDIT.md` — all intersections PASS
- A→v3 load: 242 keys, 28 new missing
- 4-step smoke: OK；Syn Dev eval≈25.68 PSNR @ step4
- Full train launched: `/root/data/experiments/ASRT/joint_syn_v3/`

```bash
python -u scripts/train.py \
  --config configs/joint_syn_v3.yaml \
  --out_dir /root/data/experiments/ASRT/joint_syn_v3 \
  --pretrained /root/data/experiments/ASRT/stage_a/checkpoints/best.pt
```
