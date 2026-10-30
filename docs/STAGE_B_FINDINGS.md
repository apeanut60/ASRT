# Stage B Findings（含官方 Test 度量 + 肉眼观感）

> 后续干净划份 + Query 对齐见 **[STAGE_B_V2_FINDINGS.md](STAGE_B_V2_FINDINGS.md)**。


更新时间：2026-10-21  
权重：Stage A `best` @58k；Stage B `best` @56k（训练中断于 ~74k，非早停）  
评测：LOLv2-Real **官方 Test n=100**；可视化：`/root/data/experiments/ASRT/eval_official_test/cases/`

---

## 1. 数字结论

| Ckpt / Mode | Test PSNR | Test SSIM |
|-------------|-----------|-----------|
| A best · bypass | **20.22** | **0.826** |
| B best · bypass | 19.89 | 0.819 |
| B best · Correct Nano | **20.50** | 0.821 |

- B+Nano 相对 A：**+0.28 PSNR**（有参考时的平均增益很小）。
- B bypass 相对 A：**−0.33 PSNR**（无参考能力略掉）。
- Dev（内部划份、带 Ref）峰值约 **19.91**，长期低于 Stage A Dev（无 Ref、不同划份）的 ~21.4，**不可直接横比**。

---

## 2. 肉眼总体观感（Test 逐图 panel）

浏览方式：Cursor 直接打开  
`eval_official_test/cases/<id>/panel.jpg`  
列顺序：Low | A-bypass | B-bypass | B-correct | NanoRef | GT。

**多数 case：**
- A / B 都能明显提亮，结构可读，**没有大面积花屏或训练崩溃迹象**。
- B-correct 相对 A：**局部纹理、对比有时更好**，但不是系统性碾压；很多图三者观感接近。
- Nano Ref 质量参差：有的接近自然白天，有的色偏/过锐，会直接影响 B-correct。

**增益最大的几张（Δ PSNR 高，观感也明显）：**  
如 `low00748`、`low00699`、`low00698`、`low00788`、`low00747`——B-correct 往往更清、更贴 GT，参考在这些图上是真的帮忙。

**掉点最大的几张：**  
如 `low00716`、`low00735`、`low00712` 等——数字上 B-correct 远低于 A，但**观感并不一定更差**（见下）。

---

## 3. 代表性 case（肉眼）

### `low00690`（射箭馆草地）— 参考伤害，肉眼可见

- A / B-bypass：草地完整、绿色连贯。
- **B-correct：大块草地被抠成灰黑洞、边缘亮绿描边**——错误纹理迁移，一眼能看出翻车。
- 同图 B-correct PSNR 还低于 B-bypass → 参考分支在此有害。
- 机制判断：极暗草地 LTE 特征弱 + hard patch 匹配，把 Ref 上不该来的块灌进 Fine。

### `low00716`（雕塑/砖墙）— 观感好、分数低

- 肉眼：B-correct **结构、材质、玻璃反光都很接近 GT**，不像 690 那种明显坏图。
- 分数：A **27.3** vs B-correct **20.8**（Δ≈−6.5），看似矛盾。
- 原因：**整图偏亮约 +18～20 灰阶**；人眼适应曝光，PSNR 重罚全局偏置。做 GT mean-align 诊断后 PSNR 可回升到 ~25.6（正式协议禁止 mean-align）。
- 结论：716 是「看起来接近、度量打曝光」；不是结构失败。

### 其余多数图

- 未再发现第二例像 690 那样「整块语义被扣掉」的严重伪影（抽查 top/worst 列表时）。
- 更常见的是：色温偏差、轻微发灰/发雾、对比与 GT 不完全一致——观感可接受，PSNR 有涨有跌。

---

## 4. 实现侧（和观感对应）

- 匹配：Raw Low / Nano → `[0,1]` → 冻结 VGG-LTE → 全局余弦 + hard argmax；**无 Gamma / HistMatch**。
- 暗区匹配脆弱 → 解释 690 草地错迁。
- Inject 软初始化 ×0.01、全程混训 Real+Syn、~25% noref。
- **P0 数据问题**：Stage B Dev 与 noref train 有大量 ID 重叠；内部 Dev 乐观，但官方 Test 不受此泄漏影响。

---

## 5. 综合判断

1. **参考通路是通的**：官方 Test 上 Correct Nano 略高于 A；肉眼在部分图上能看到纹理/清晰度收益。  
2. **收益不稳定、幅度小**：平均 +0.28 dB；无参考路径还变差；存在 690 类灾难 case 与 716 类「观感≠PSNR」case。  
3. **当前瓶颈不像「完全没注入」**，更像：**暗光匹配不可靠 + 照明/曝光漂移 + 参考质量不均**。  
4. Stage C（Huawei/Nikon ~500 Nano FT）前，更值得先修匹配前亮度对齐、Val 隔离、以及参考失败时的 bypass 回退；否则域自适应可能放大坏迁移。

---

## 6. 产物路径

| 内容 | 路径 |
|------|------|
| Test 汇总 | `eval_official_test/SUMMARY.json` |
| 逐图 panel | `eval_official_test/cases/<id>/panel.jpg` |
| 增益排序索引 | `eval_official_test/cases_index.json` |
| 实现审计 | `ASRT_STAGE_B_IMPLEMENTATION_AUDIT.md` |
