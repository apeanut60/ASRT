# ASRT-v3 Architecture

完整可靠性引导参考恢复网络（一次性全量实现，不做分阶段消融）。

## 保留自 B-v2

- Query Gamma + Histogram Alignment（仅 LTE Query）
- Frozen VGG-LTE + SearchTransfer 纹理提取
- CSFI-U-Net Coarse / Middle / Fine + RGB Light-up + MergeTail
- 干净 `manifests_b_v2` Train/Dev

## 新增模块

### 1. Dual Reliability Estimator (`asrt/models/reliability.py`)

SearchTransfer 额外诊断（复用相似度矩阵 R）：

| 信号 | 含义 |
|------|------|
| `match_index` | Top-1 Key 索引（H/4 flattened） |
| `match_score` | Top-1 cosine |
| `match_margin` | Top-1 − Top-2 |
| `cycle_consistency` | 双向一致性（辅助，非硬拒绝） |
| `spatial_consistency` | 邻域位移连续性 |

叠加 Low 结构图（梯度幅值、局部方差，经分位数软归一）。轻量 CNN → **`C_T`, `C_L`** ∈ `[0,1]` @ H/4。  
初始化：`C_T` bias=+2（≈0.88，近开放注入）；`C_L` bias=0（≈0.5），照明效果由零初始化 Adapter 控制。

Fine：`FineCTRefine` 用高分辨率 Low 结构轻量修正上采样 `C_T`（硬 0/1 透传）。

### 2. Reliability-aware Texture Injection

```text
F_new = F + C_T_scale * σ(S) * Head([F∥T])
```

三尺度均启用。`C_T=0` ⇒ 参考纹理残差严格为 0。保留原 Head 权重。

### 3. Matched Reference Illumination Prior (`illumination_prior.py`)

```text
P_ref = LPF(luminance(NanoRef))     # 低频亮度代理，非物理 illumination
P_ref_matched = warp(P_ref, match_index) @ H/4 + 低频平滑
P_base = LPF(luminance(lightup_low))
D_ref = log(P_ref_m+ε) - log(P_base+ε)   # 去掉全局均值，仅局部差
```

**不用 GT。** Fine 不注入参考照明。

### 4. Illumination Adapter → Reference-guided IGAB

```text
L_guided = L_low + C_L * DeltaL_ref
V_mod = V * L_guided          # 仅替换原 V*L_low；IG-MSA 其余不变
```

Coarse / Middle 各一 Adapter；末层零初始化。`C_L=0` ⇒ 无参考照明修正。

### 5. Reference-conditioned ResBlock (`cond_resblock.py`)

Coarse `RB11[-1]`、Middle `RB22[-1]`：

```text
H = ReLU(Conv1(F))
H_mod = H * (1 + C_L*γ) + C_L*β     # γ,β = Adapter(L_guided)，末层零初始化
F_out = F + Conv2(H_mod)
```

`ref=None` 或不传条件 ⇒ 原 ResBlock。Fine 不做照明调制。

### 6. No-reference Preservation

- `ref=None`：跳过 LTE / ST / Reliability / Illum Prior / Cond / 注入
- 保持 `no_ref_prob` dropout
- 无参考 batch：`L = L1(pred, gt) + λ L1(pred, stopgrad(Teacher_A))`，`λ=0.1`（初值）
- Teacher = Stage A 58k，frozen eval

## 数据流（有参考）

```text
Raw Low + Nano Ref
  → Query Align → LTE(Q) ; LTE(Ref)→K,V
  → SearchTransfer + diag
  → DualReliability → C_T, C_L
  → Matched Illum Prior (match_index warp)
  → Inject(C_T) / CondRB / IGAB(L_guided) on Raw-Low backbone
```

## 初始化

从 `stage_b_v2/checkpoints/best.pt`（70k）严格加载旧键；仅允许新增模块缺失。新 Optimizer / Scheduler / GradScaler；`stable_steps=2000` 仅训 v3 新模块。

## 未改

- IGAB 注意力实现（仍为通道注意力，非 HW×HW）
- CSFI 跨尺度拓扑、Loss 主项 L1
- 官方 Test 不参与选模 / 调阈值
