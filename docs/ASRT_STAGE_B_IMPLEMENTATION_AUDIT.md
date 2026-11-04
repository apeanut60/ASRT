# ASRT Stage B Implementation & Protocol Audit

**Audit time:** 2026-10-21 (read-only)  
**Running process:** PID 9452 — `python -u scripts/train.py --config configs/stage_b_lolv2_reference.yaml --out_dir /root/data/experiments/ASRT/stage_b --resume .../step_006000.pt`  
**Observed progress at audit:** ~step 68400–68550 / 100000  
**Constraint:** No code/config/weight/data/process modifications were made.

---

## A. Executive Summary

Current Stage B trains ASRT with frozen ImageNet VGG19-LTE + parameter-free SearchTransfer, injecting hard-matched Nano-Ref features into CSFI-UNet at H/4, H/2, H. Matching Query is **Low RGB** mapped to `[0,1]` then ImageNet-normalized; **no Gamma / Histogram Matching / CLAHE** before LTE. Key/Value come from the same Nano Ref RGB (also `[0,1]` + MeanShift). Stage A warm-start used `stage_a/.../best.pt` at **step 58000** (Dev PSNR ~21.43); `load_state_dict` is **strict** and matches all keys. Live recipe after restart: **batch=8, accum=1**, Real+Syn triplets (~75% ref) + Real+Syn paired noref (~25%), L1, AMP FP16, grouped Adam LRs. **P0:** Stage B Dev IDs largely appear in the **no-ref paired train** loader (30/32), so validation images can still receive Low/GT gradient via noref batches. Soft inject init (×0.01) and Stage B Dev PSNR (~19.9 with ref) vs Stage A (~21.4 no-ref, different split) are not directly comparable.

---

## B. Actual Reference Matching Pipeline

### [实际代码中的 Query 数据流]

1. Disk PNG → `read_rgb_m11` → tensor **[-1, 1]**  
   Evidence: `asrt/data/paired.py:13-16` (used by `asrt/data/triplet.py:10,38`).
2. Train batch → `Trainer.train` passes `low` to `ASRT.forward`  
   Evidence: `asrt/training/trainer.py:207-212`.
3. If `ref is not None`: `match_reference(low, ref)`  
   Evidence: `asrt/models/asrt.py:78-83`.
4. Pad Low to multiple of 4 (`safe_pad`); **no gamma/hist**  
   Evidence: `asrt/models/asrt.py:59-63`.
5. `q01 = (low_p + 1) * 0.5` → **[0, 1]**  
   Evidence: `asrt/models/asrt.py:64`.
6. `_, _, q_lv3 = self.lte(q01)` — LTE/VGG on Query only at **lv3**  
   Evidence: `asrt/models/asrt.py:66`.
7. LTE: ImageNet `MeanShift` then VGG slices → lv1/lv2/lv3  
   Evidence: `asrt/models/reference.py:42-50`, slices `0:2`, `2:7`, `7:12` (`reference.py:29-34`).

**Not used for Query matching:** IlluminationEstimator / light-up gain (those run only inside backbone on Low). Evidence: light-up at `csfi_unet.py:264-270`; matching path never calls it (`asrt.py:55-70`).

**Keyword search (asrt/ package, Stage B path):**

| Keyword | Result |
|---------|--------|
| `gamma` | Only in **disabled** `GlobalIllumHead` (`legacy_heads.py:27-44`); Stage B `use_global_illum: false` (`stage_b_*.yaml:29`) |
| `hist` / CLAHE / equaliz | **No matches** in `asrt/` train/model/data |
| `LTE` / `SearchTransfer` | `asrt.py:44-45,66-69`; `reference.py` |
| `illum` | Backbone only (`csfi_unet.py:186-188,264-280`) — not matching preprocess |
| `RefCorrection` | **Absent** in `asrt/`; only mentioned in `legacy_prototype/.../warmstart_csfi_unet.py` comment |

### [实际代码中的 Key 数据流]

1. Nano Ref PNG → same `read_rgb_m11` → **[-1, 1]** (`triplet.py:41-42`).
2. Resize to Low HW + pad ×4 (`asrt.py:57-63`).
3. `r01 = (ref_p + 1) * 0.5` (`asrt.py:65`).
4. `_, _, k_lv3 = self.lte(r01)` (`asrt.py:68`) — **third LTE forward on same Ref** as Values’ `ref_lv*`.
5. No RefCorrectionHead; K is LTE(lv3) of raw Ref RGB in [0,1] + MeanShift.

### [实际代码中的 Value 数据流]

1. Same Nano Ref `r01` (`asrt.py:65-67`): `ref_lv1, ref_lv2, ref_lv3 = self.lte(r01)`.
2. Channels: lv1=64 (H), lv2=128 (H/2), lv3=256 (H/4) after VGG pools (`reference.py` + VGG indices 0–11).
3. Unfolded and gathered by match arg in `transfer_from_arg` (`reference.py:64-79`).

### [SearchTransfer 的真实数学操作]

Evidence: `asrt/models/reference.py:81-89`.

- Unfold Q/K patches **3×3**, L2-normalize along feature dim.
- `R = bmm(K, Q)` → dense similarity over all ref patches × all query patches (**global** search).
- `R_star, R_arg = max(R, dim=1)` → **hard argmax** (not softmax / top-k / learnable).
- `S = R_star` reshaped `[B,1,H/4,W/4]` — cosine similarity in **[-1,1]** theoretically; observed example ~[0.68, 0.84] on random tensors (runtime Stage B gate stats: **UNVERIFIED** — no diagnostic log).
- Values transferred by gather+fold at native resolutions; **no Identity Transfer** branch in code.

Shapes (H=W=64 example): `S [B,1,16,16]`, `T3 [B,256,16,16]`, `T2 [B,128,32,32]`, `T1 [B,64,64,64]` — verified in audit probe. For crop 256: H/4=64 → `T3 [B,256,64,64]`, etc.

**Historical B2/A0/Identity:** **No** such branches in `asrt/models/`. Only soft residual inject (`feat + g * head([feat,T])`).

### [最终 S/T 的去向]

`ASRT.forward` → `backbone(..., S, T_lv3, T_lv2, T_lv1, use_reference=True)` (`asrt.py:85-88`).

- Coarse H/4: `_inject(e2, T_lv3, S, conv11_head)` (`csfi_unet.py:278`)
- Middle H/2: `_inject(x22, T_lv2, S, conv22_head)` (`csfi_unet.py:284`)
- Fine H: `_inject(x33, T_lv1, S, conv33_head)` (`csfi_unet.py:292`)
- Gate: `sigmoid(bicubic_upsample(S))` (`csfi_unet.py:217-221,227-228`)

### Final one-line (Part 1)

**当前 Stage B 是否在没有 Gamma + Histogram Matching 的条件下，直接使用 Raw Low 与 Nano Ref 的 LTE 特征做纹理匹配？**

**YES**

Evidence: load path `paired.py:13-16`; match path `asrt.py:64-69` only does affine `[-1,1]→[0,1]` + ImageNet MeanShift inside LTE (`reference.py:42-43`); no hist/gamma on matching inputs; Grep found no hist/CLAHE in `asrt/`.

---

## C. Actual Network Structure

```text
Low[-1,1]
  ├─(match only if ref)─► [0,1] ► LTE ► q_lv3 ─┐
  │   Ref[-1,1] ► [0,1] ► LTE ► k_lv3, ref_lv* ─┼► SearchTransfer ► S,T3,T2,T1
  │                                              └──────────────────────────┐
  ├─ pad×4 ► IllumEstimator(low01) ► illu_h, log_gain                      │
  │            enable_lightup=True ► Y_base = 2*(low01*exp(gain))-1         │
  ├─ SFE(e0) ► down1(e1) ► down2(e2)                                        │
  │     e2 + inject(T3,S) ► RB11 ► coarse_igab(·, illu_h4)                  │◄─┘
  │     skip+e1 + inject(T2,S) ► CSFI2(ex12) ► RBs ► middle_igab(·, illu_h2)
  │     skip+e0 + inject(T1,S) ► CSFI3(ex123) ► RBs
  └─ MergeTail residual + Y_base ► out
       use_ref_illum=False, use_global_illum=False  (both OFF)
```

- **CSFI2/CSFI3:** bidirectional cross-scale merge (`csfi_unet.py:59-100,285,293`).
- **IGAB illum:** resized features from **Low-only** IlluminationEstimator (`csfi_unet.py:275-280,288`) — **not** Ref-conditioned.
- **Light-up:** restoration path only (`csfi_unet.py:266-268`), distinct from matching preprocess.
- **Inject soft-init:** head weights ×0.01, bias 0 (`csfi_unet.py:180-184`).

---

## D. Actual Training Recipe

Sources: live cmdline + `configs/stage_b_lolv2_reference.yaml` + `scripts/train.py:46-50` + `trainer.py` + manifests + log.

| Item | Actual value | Evidence |
|------|--------------|----------|
| Total steps | 100000 | yaml:4-5; log `.../100000` |
| Current step (audit) | ~68550 | `train.jsonl` last line |
| batch_size | **8** (after restart); was 1 for steps 0–6000 | yaml:6; log L131; ckpt cfg |
| accum | 1 | yaml:7; log |
| effective batch | 8 | log `effective_batch=8` |
| crop (ref loader) | **256 for Real and Syn triplets** | `trainer.py:160` uses `crop_ref` only |
| crop_syn | 128 — **only noref Syn loader** | `trainer.py:154-156` |
| Optimizer | Adam, grouped | `optimizer.py:17-52` |
| lr_backbone | 1e-5 | yaml:9 |
| lr_inject | 1e-4 | yaml:10 |
| lr_igab | 2e-5 | yaml:11 |
| lr_illum (+ merge_tail) | 2e-5 | yaml:12 |
| Scheduler | `LambdaLR` cosine to `eta_min/lr` ratio over `total_steps` | `scheduler.py:9-24` |
| eta_min | 1e-6 | yaml:13 |
| Loss | L1 only, weight 1 | `trainer.py:23-24,213` |
| Grad clip | 1.0 | yaml:15; `trainer.py:220-224` |
| AMP | GradScaler + autocast (FP16), CUDA | yaml:14; `trainer.py:34,211` |
| BF16 | No | — |
| EMA | No | no EMA in trainer |
| Seed | 42 | yaml:3 |
| Augment | sync crop/flip/rot90 on Low/GT/(Ref) | `paired.py:19-38`; `triplet.py:57-61` |
| Mixup | **No** | no mixup in asrt/ |
| no_ref_prob | 0.25 | yaml:18; `trainer.py:197` |
| log/eval/save | 50 / 2000 / 2000 | yaml:19-21 |
| Best criterion | max Dev `psnr_rgb` | `trainer.py:255-257` |
| Eval set | `lolv2_real_triplet_dev.json`, `eval_limit=32` | `train.py:49`; yaml:22 |
| Eval with ref | **Yes** (stage≠a and rows have ref_path) | `trainer.py:250-251` |
| Metrics | RGB PSNR/SSIM on uint8; **no** GT mean-align | `metrics.py:1,22-59` |
| Workers | 2 | yaml:23 |

### Dataset counts (manifests under `/root/data/experiments/ASRT/manifests/`)

| Split | n | Role |
|-------|---|------|
| lolv2_real_triplet_train | 607 | Ref train (Real∩Nano) |
| lolv2_syn_triplet_train | 238 | Ref train (Syn∩Nano) |
| lolv2_real_paired_train | 655 | Noref train Real |
| lolv2_syn_paired_train | 855 | Noref train Syn |
| lolv2_real_triplet_dev | 32 | Stage B eval |
| Roots | Real Low/GT/`nanobanana_ref_v2`; Syn Low/GT/`nanobanana_ref` | yaml data: + audit |

All triplet train rows have non-null `ref_path` (verified). Missing Nano excluded at manifest build (`manifest.py:70-72`), not filled with GT (`triplet.py:23-28`).

**Sampling:** each step, with prob `1-no_ref_prob=0.75` take ref loader (Real+Syn triplets shuffled together); else noref_real or noref_syn with equal random choice among noref keys (`trainer.py:196-206`). Exact Real:Syn ratio inside ref loader is DataLoader-shuffle dependent — **UNVERIFIED** as a fixed fraction.

---

## E. Stage A Weight Transfer

| Item | Finding | Evidence |
|------|---------|----------|
| Initial load path | `/root/data/experiments/ASRT/stage_a/checkpoints/best.pt` | log L1; yaml:24; first launch cmdline historically used `--pretrained` |
| Checkpoint step | **58000** (not 80k last) | `best` meta: step 58000, PSNR 21.427 |
| File SHA256 | `1d25b83a664fcb4311b50bf94ce256f52fe26a5fd0cd10bebff5ccdbbf106afa` | audit hash |
| Load API | `load_state_dict(ck['model'])` default **strict=True** | `trainer.py:77` |
| Missing/unexpected | None on strict reload into current ASRT | audit probe |
| Restart | At step 6000 resumed `step_006000.pt` with bs=8 | log L129-131 |
| Frozen | LTE (`freeze_lte: true`) | yaml:28; `asrt.py:44`; `reference.py:35-37` |
| SearchTransfer | **0 parameters** — N/A train/freeze | audit |
| Trainable in optim | backbone trunk, inject heads, IGAB, illum+merge_tail | `optimizer.py:17-52`; no leftover trainable params |
| New ref modules | Inject heads present in Stage A ckpt (soft-init from A); LTE ImageNet | keys include `backbone.conv11_head.*`, `lte.*` |
| `strict=False` | Only in `scripts/evaluate.py:42` — **not** Stage B train load | |

**Param groups (fresh Stage B cfg on A weights):**

| Group | LR | #params |
|-------|-----|---------|
| backbone | 1e-5 | 3,251,756 |
| inject | 1e-4 | 368,832 |
| igab | 2e-5 | 105,602 |
| illum_tail | 2e-5 | 131,398 |
| LTE frozen | — | (non-trainable) |
| **Trainable total** | | **3,857,588** |

---

## F. Potential Implementation Risks

### P0

1. **Val ID leakage via noref paired train**  
   - `lolv2_real_triplet_dev` ∩ `lolv2_real_paired_train` = **30 / 32** IDs (same `low_path`).  
   - Stage B still samples noref with p=0.25 (`trainer.py:197`), so Dev images can be optimized as Low/GT without Ref.  
   - Trigger: Stage B training as currently configured.  
   - Evidence: manifest ID sets (audit); `train.py:47-49`.

2. **Stage A Dev images enter Stage B ref train**  
   - `lolv2_real_paired_dev` ∩ `lolv2_real_triplet_train` = **28** IDs.  
   - Not classic val leak for Stage B metric, but Stage A’s 32-eval pool is further trained in B.  
   - Evidence: manifest intersection audit.

### P1

1. **Syn triplet crops are 256, not 128**  
   - `crop_syn: 128` only applies to **noref** Syn (`trainer.py:154-160`); ref Syn uses `crop: 256`.  
   - Differs from Retinexformer-style Syn-128 expectation in brief.

2. **Soft inject ×0.01** may keep early ref residual tiny (`csfi_unet.py:180-184`); combined with hard match on dark Low features, may slow ref learning — behavioral impact **partially evidenced** by Stage B Dev PSNR plateau ~19.9 (with-ref protocol) vs Stage A ~21.4 (no-ref, different split) — comparison confounded.

3. **Cosine schedule min_ratio uses scalar `cfg['lr']`** (`scheduler.py:12-14`), so inject group floor ≠ `eta_min` in absolute LR.

4. **High clip rate** — jsonl `clip_events` ≈ 62550 at step 68550 (most steps clip); may indicate unstable grads with ref path.

5. **LTE called thrice on Ref** (`asrt.py:67-68`) — correct functionally (same graph), wasteful; K≡ features of same forward as V lv3 up to randomness none.

### P2

1. No logging of `missing_keys` on load (strict success is silent).  
2. No gate/S/grad diagnostics in train log.  
3. `evaluate.py` uses `strict=False` — future eval footgun, not current train.  
4. Ref loader mixes Real+Syn without explicit domain balancing.

---

## G. Unverified Items

- Live per-step Gate mean/std / whether Gate saturates near 0 or 1 during Stage B.
- Non-zero gradients through inject heads on a **running** step (would require hooks; not done).
- Exact Real vs Syn sampling fraction inside the combined ref DataLoader.
- Whether NVIDIA reserved memory equals activation need (log peak ~6661 MB after bs=8).
- Official LOLv2 Test metrics for Stage A/B checkpoints (not run in this audit).
- Content of Nano Ref images beyond path naming (not opened).

---

## H. Final Answers

```text
Q1. SearchTransfer Query 是否来自 Raw Low：YES
Q2. 是否启用 Gamma Correction：NO
Q3. 是否启用 Histogram Matching：NO
Q4. K/V 是否来自 Nano Ref：YES
Q5. 是否存在额外 RefCorrection：NO
Q6. 是否使用全局 Patch Matching：YES
Q7. 是否使用 Identity Transfer：NO
Q8. Stage A 主干是否成功加载：YES
Q9. Stage B 是否混合无参考 Batch：YES
Q10. IGAB 是否使用 Ref 照明信息：NO
Q11. 各级参考特征是否实际注入恢复网络：YES
Q12. Train/Eval 是否具有一致的照明与输出协议：YES
Q13. Stage B Train/Val 是否严格隔离：NO
```

### Notes on H answers

- **Q1:** Raw Low file → `[-1,1]` → match uses `(x+1)/2` only; not light-up / not hist.  
- **Q2:** Matching path has no gamma; `GlobalIllumHead` gamma exists but **OFF**.  
- **Q12:** Same `forward` / light-up flags for train and eval; eval uses full-image (or tiled if side>512) without GT align (`trainer.py:94-132`, `metrics.py`). Train uses random crops — geometry differs by design, not illum protocol.  
- **Q13:** **NO** due to noref paired_train ∩ triplet_dev (P0). Triplet train vs triplet_dev overlap is 0.

---

*End of audit. No files other than this report were written; training process was not interrupted.*
