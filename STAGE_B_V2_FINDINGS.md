# Stage B-v2 Findings

更新时间：2026-10-22  
权重：Stage A `best` @58k → Stage B-v2 从零开训；B-v2 `best` @**70k**（满跑 100k，总时长 ~1137 min）  
评测：LOLv2-Real **官方 Test n=100**；内部 Ref Dev **n=30**（干净划份）  
协议：RGB PSNR/SSIM，`crop_border=0`，**禁止 GT mean-align**

相对 Stage B-v1，本轮只改了两件事：**(1) Query Gamma+Histogram 对齐**；**(2) Train/Dev 泄漏修复并全新重训**。CSFI / IGAB / Gate / Loss / 主干结构未动。

---

## 1. 数字结论（官方 Test）

| Ckpt / Mode | Test PSNR | Test SSIM |
|-------------|-----------|-----------|
| A best · bypass | **20.22** | **0.826** |
| B-v1 best · bypass | 19.89 | 0.819 |
| B-v1 best · Correct Nano | **20.50** | 0.821 |
| B-v2 best · bypass | 19.96 | 0.818 |
| **B-v2 best · Correct Nano** | **20.20** | **0.825** |

相对关系（同一官方 Test）：

| 对比 | Δ PSNR |
|------|--------|
| B-v2+Nano − A bypass | **−0.02**（基本持平，未超过 A） |
| B-v2+Nano − B-v2 bypass | **+0.24**（参考有正增益，但幅度小） |
| B-v2+Nano − B-v1+Nano | **−0.30**（低于旧 B；条件不同，见 §4） |
| B-v2 bypass − A bypass | **−0.26**（无参考能力仍弱于 A） |

**一句话：** Query 对齐 + 干净划份后，参考通路仍通，但官方 Test 上 **未能超过 Stage A**，也 **未超过旧 B-v1+Nano**；平均参考增益从 B-v1 的 ~+0.61（相对自身 bypass）缩到 ~+0.24。

---

## 2. 内部 Dev（干净划份，n=30 Ref）

| 节点 | Dev PSNR | Dev SSIM |
|------|----------|----------|
| 2k（起步） | 20.80 | 0.750 |
| **70k（best）** | **21.33** | 0.790 |
| 84k | 21.31 | 0.793 |
| 96k | 21.25 | 0.795 |
| 100k（last） | 21.22 | 0.795 |

- 曲线：前段上升 → 中后期在 **21.0–21.3** 震荡；PSNR 峰在 **70k**，之后略回落，SSIM 仍缓升。
- 与 Stage A Dev（同划份、无 Ref，峰值 ~**21.43**）相比：B-v2 Ref Dev best **21.33**，已接近但不越过 A。
- **勿与 B-v1 内部 Dev（~19.91）横比**：B-v1 Dev 与 noref train 大量重叠，且样本集合不同。

完整 34 张无参考 Dev 未作为选模依据；选模仅看 **n=30 Ref Dev**。

---

## 3. 本轮改了什么、没改什么

### 已落地

1. **Query Illumination Alignment**（仅匹配用）  
   - Raw Low + Nano Ref → 整图估 Gamma / Hist LUT → 只改 Query  
   - Raw Nano → K/V；Raw Low → 恢复主干  
   - 训练 Crop / 推理 Tile **复用同一组全图统计**  
   - 参数：`α=0.75`，`β=0.50`（沿用 TTSR-lowlight 可恢复的有效配方，非任务文中的 fallback `β=0.20`）

2. **Train/Dev 隔离**  
   - 新 Manifest：`/root/data/experiments/ASRT/manifests_b_v2/`  
   - Canonical Dev = Stage A paired Dev（n=34）；Ref Dev = 有 Nano 子集（n=30）  
   - 全部 train loader 排除这些 ID；交集审计全部 PASS  
   - **从 Stage A 58k 严格重载重训**，不继承 B-v1 优化器 / 权重

### 未改

- CSFI-U-Net、IGAB、Gate、L1、LTE 冻结、分组 LR、`no_ref_prob=0.25`、bs=8、100k  
- **未做** Reference-conditioned IGAB  
- **未做**「同干净 Split 上关对齐」的正式消融（配置已备：`configs/stage_b_v2_baseline_no_align.yaml`，未跑满）

---

## 4. 如何解读「比 B-v1 还低」

不能把 B-v2 Test 20.20 vs B-v1 Test 20.50 直接读成「对齐有害」：

1. **B-v1 训练数据曾接触旧 Dev ID**（泄漏）；B-v2 把这些图踢回/隔离后，训练分布变了。  
2. **选模 Dev 不同**：B-v1 best@56k 在泄漏 Dev 上选；B-v2 best@70k 在干净 Ref Dev 上选。  
3. **对齐与泄漏修复同时引入**，缺少「仅改一项」的对照；要对齐机制独立收益，必须跑 `stage_b_v2_baseline_no_align`。

因此当前只能说：**在干净协议下，现配方的参考增强平均收益偏小，且未稳定超过 Stage A。**

---

## 5. 伪影（肉眼主结论）

详见 **[STAGE_B_V2_ARTIFACT_FINDINGS.md](STAGE_B_V2_ARTIFACT_FINDINGS.md)**。摘要：

- **有结构区**：文字/边缘可正确从 Nano 迁纹理（如 `low00733`）。  
- **极暗大面积纯色区**：不规则杂色斑普遍；来自 hard match 误注入，非主干单独崩溃。  
- Correct vs bypass：**51/100 掉分**；733 类「局部观感赢、整图 PSNR 输」。  
- Query 对齐不解决「该不该注入」；优先置信度/平坦区门控，而非更强全局亮度对齐。

可视化：`eval_official_test_b_v2/cases/`（690 草地错迁仍在）。

---

## 6. 综合判断

1. **参考通路工作正常**：Test 上 Correct > bypass（+0.24）；Dev 上带 Ref 可到 ~21.3；肉眼可见正确文字迁移。  
2. **平均收益仍不够**：Test 几乎打平 Stage A；无参考路径继续掉点；一半样本参考有害。  
3. **主瓶颈是暗平坦区 hard match 伪影 / 错迁**，不是「没注入」。  
4. Query 对齐的独立收益仍缺消融；伪影修复应优先 **失败抑制**，再考虑 Ref-conditioned IGAB / Stage C。

---

## 7. 产物路径

| 内容 | 路径 |
|------|------|
| 训练目录 | `/root/data/experiments/ASRT/stage_b_v2/` |
| best / last | `.../checkpoints/best.pt`（70k）、`last.pt`（100k） |
| 训练日志 | `.../stage_b_v2/train.stdout.log` |
| Test 结果 JSON | `eval_official_test/stage_b_v2_best_test_both.json` |
| Test 可视化 | `eval_official_test_b_v2/` |
| 伪影 findings | `docs/STAGE_B_V2_ARTIFACT_FINDINGS.md` |
| Manifest / 审计 | `manifests_b_v2/`、`STAGE_B_V2_SPLIT_AUDIT.md` |
| Changelog / Smoke | `STAGE_B_V2_CHANGELOG.md`、`STAGE_B_V2_SMOKE_TEST.md` |
| B-v1 肉眼 findings | `docs/STAGE_B_FINDINGS.md` |
