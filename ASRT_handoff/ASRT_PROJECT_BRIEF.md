# ASRT — 新仓库项目规格书（2026-10-20）

> **项目定位**：ASRT（独立模型/仓库名称）；方法描述：*Illumination-Guided Cross-Scale Reference Restoration for Low-Light Image Enhancement*。ASRT 暂不强行反向扩写缩写。  
> **建议 GitHub 仓库**：`apeanut60/ASRT`（建议名称，尚未创建；论文命名仍可调整）  
> **状态**：模型结构已有独立原型与 CPU smoke tests；**尚未完成无参考训练接口、真实数据训练及性能验证**。本文是给代码助手的工程执行规格，不是论文实验结果。

## 0. 决策与范围

目标是从旧 `TTSR-lowlight` 中**独立建立一个新项目**，而不是在旧仓库继续堆版本。

- **采用完整主干改造，不先做模块消融。** 先完成可训练的端到端 v1，后续才安排消融。
- **保留研究资产**：LTE/VGG 特征提取、SearchTransfer、原三级参考转移、TTSR 的 CSFI 思想、已完成的诊断/数据经验；新仓库独立命名，旧仓库只读作为历史基线。
- **吸收 Retinexformer 的部分机制而非整网换成 Retinexformer**：IGAB（illumination-guided channel-transposed attention + convolutional FFN）、照明估计与残差恢复；不对外宣称“Retinexformer 预训练权重可直接迁移”。
- **核心研究课题**：在真实低光退化和有限生成参考约束下，让跨尺度参考信息成为恢复的可靠补充。未来可能研究可靠的跨退化结构引导（暂称 A2），**但它尚不属于当前已实现的 v1 方法贡献**。ASRT 是项目名称，不代表可靠性机制已完成。
- **不能靠新增几千张 Nano Banana 参考**；禁止偷偷把 GT 作为 Nano 的替代品。

## 1. 原型与外部实现依据

### 1.1 可复用资产

- 原始仓库：`https://github.com/apeanut60/TTSR-lowlight`，已核对分支 `experiment/ttsr-lowlight-lolv2real`。
- 可用原型压缩包：本会话的 `ttsr_csfi_unet_v3.zip`，含 `model/MainNetCSFIUNet.py`、安装脚本、旧权重迁移脚本和 7 项基础测试。
- 原型的 `MainNetCSFIUNet.py` 在本地重新执行 7 项基础测试均通过，**只代表基础形状/梯度/安装正确，不能代表完整训练可运行或 PSNR 达标**。
- 原版 TTSR `MainNet.py`：`https://github.com/researchmm/TTSR/blob/master/model/MainNet.py`。
- Retinexformer 官方 IGAB/IG-MSA：`https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/models/archs/RetinexFormer_arch.py`。
- Retinexformer 官方配置：
  - `https://github.com/caiyuanhao1998/Retinexformer/blob/master/Options/RetinexFormer_LOL_v2_real.yml`
  - `https://github.com/caiyuanhao1998/Retinexformer/blob/master/Options/RetinexFormer_LOL_v2_synthetic.yml`
- 新仓库保留 LICENSE、源代码引用/修改说明，先核验各来源许可证约束。

### 1.2 模型与命名

- 项目名/仓库名：**ASRT**。不沿用旧 TTSR 系列名；CSFI-U-Net 是内部主干结构名。
- 网络类建议：`ASRT`，核心恢复类 `CSFIUNet`；不要再将模型称作旧 `MainNetEnhance`。
- 方法描述：*Reference-guided Low-light Enhancement with Illumination-aware Cross-scale Restoration*。
- `CSFI-U-Net` 是主干的结构名，`ASRT` 是整个可训练模型名。

## 2. 架构总览（当前达成共识）

```text
                 Low X                              Nano Ref R
                   |                                  |
      +------------+-----------+                      |
      |                        |                      v
 SFE/Encoder            Illumination Estimator    LTE / Key-Value
      |                        |                      |
 E0: H,   C=64             L_H, L_H/2, L_H/4     SearchTransfer
      |                        |                      |
 E1: H/2, C=64             log_gain            S, T_lv3, T_lv2, T_lv1
      |                        |                      |      |       |
 E2: H/4, C=64 --------> Coarse H/4 <---- T_lv3 -----+      |       |
                                RB11 + IGAB(L_H/4)         |       |
                                /          \               |       |
                     Coarse retained    Up + E1 Skip       |       |
                                \          /               |       |
                                 Middle H/2 <--- T_lv2 ----+       |
                                 CSFI2 + RB21/RB22 + IGAB  |       |
                                     /          \                  |
                         Coarse retained  Up + E0 Skip            |
                                       \     /                     |
                                        Fine H <--------- T_lv1 ---+
                                     CSFI3 + RB31/RB32/RB33
                                         |     |     |
                                  Multi-scale Merge
                                         |
                                    RGB Residual
                                         +
                              X adjusted by light-up gain
                                         |
                           optional, explicitly controlled legacy
                           RefIllumTransfer / GlobalIllumHead
                                         |
                                  Enhanced RGB Yhat
```

### 2.1 三条流（各自天然分辨率）

输入：`Low X : [B,3,H,W]`，传统 TTSR 路径使用 `[-1,1]`；内部 Light-up 在 `[0,1]` 计算。

| 部件 | 分辨率 | 默认通道 | 源/作用 |
| --- | --- | ---: | --- |
| `E0 = SFE(X)` | H×W | 64 | 高分辨率 Encoder Skip |
| `E1 = Down(E0)` | H/2×W/2 | 64 | 中尺度 Encoder Skip |
| `E2 = Down(E1)` | H/4×W/4 | 64 | Coarse 初始恢复特征 |
| Coarse Stage 1 | H/4×W/4 | 64 | `T_lv3 (256ch)` 注入 + 8 RB + 1 IGAB |
| Middle Stage 2 | H/2×W/2 | 64 | Up + E1 Skip + `T_lv2 (128ch)` + CSFI2 + 每支 4 RB + 1 IGAB |
| Fine Stage 3 | H×W | 64 | Up + E0 Skip + `T_lv1 (64ch)` + CSFI3 + 每支 2 RB |
| MergeTail | H×W | →3 | 三尺度特征合并，预测 RGB 残差 |

默认深度 `num_res_blocks='8+8+4+2'`，合计 **30 个 ResBlock**（SFE 8，Stage1 8，Stage2 2×4，Stage3 3×2），加粗尺度/中尺度各 1 个 IGAB。

### 2.2 Reference 分支

- 沿用已有的 `LTE + SearchTransfer`：Low Query 与生成参考的 Key/Value 做 patch matching，输出 `S, T_lv3, T_lv2, T_lv1`。
- `T_lv3`、`T_lv2`、`T_lv1` 不再全部 Bicubic 上采样至 H×W，而是在 H/4、H/2、H 的恢复 Stage 原生注入。
- 参考注入约为 `F' = F + gate(S) * Conv(cat(F, T_lv*))`；对未训练的新注入层提供**温和初始化/可控残差幅度**，避免一接参考就破坏已学的恢复。
- CSFI2 是粗/中尺度**双向**交换；CSFI3 是粗/中/细之间**两两**交换。与普通只沿 Encoder→Decoder 单向融合的 U-Net 不同。
- 匹配分数 `S` **不是生成内容可靠性的充分证据**，禁止在文稿中把现有 `sigmoid(S)` 宣称为已解决幻觉检测。

### 2.3 Illumination + IGAB

- `IlluminationEstimator`: 低光图 + 单通道 mean，输出 H 分辨率的照明特征与 3ch `log_gain`；照明特征按需缩放到 H/2、H/4。
- 粗、中尺度分别采用官方语义的 `IGAB`：Q/K/V；照明特征调制 V；**转置通道注意力 [head,d,d]**，不是 HW×HW 空间注意力；再加 depthwise-conv FFN。
- 光照增益用于最终 RGB 基底：`Y_base = 2 * (Low_01 * exp(log_gain)) - 1`，输出 `Y_pre = Y_base + RGBResidual`。
- **注意：当前原型没有把 Light-up 图作为 Encoder 输入**。不能把当前实现描述成完全复制 Retinexformer 的 Light-up → Denoiser 流程。
- `RefIllumTransfer`、`GlobalIllumHead` 暂保留，但应有显式开关，防止数个亮度校正头在训练时互相补偿；训练/推理必须统一。

## 3. 已发现、必须在正式训练前修复的问题

1. **完整 Reference Bypass**：原型外层 `TTSREnhance` 仍依赖 `ref` 和 `S/T`。新仓库必须支持 `model(low, ref=None)`，并跳过 LTE、SearchTransfer、三个参考注入、RefIllumTransfer；`S=None` 时诊断钩子也不能崩溃。**不得用 GT 填充 Ref，不得默认静默替代成其他图像**。
2. **返回 API 统一**：建议 `forward(low, ref=None, return_aux=False) -> enhanced_rgb`；诊断存在时返回明确定义的字典，避免旧五元组 `None` 破坏原 trainer 的 `tpl_loss`。
3. **训练级对齐检查**：Low、GT、Ref 必须同名匹配，几何变换、随机 Crop 同步。Nano 数据缺失时按采样策略走无参考分支，禁止悄悄收缩全量训练集，也不允许“缺参考就用 GT”。
4. **尺度与 padding**：在进入 LTE/SearchTransfer 与 Encoder 之前定义唯一的 Pad/Unpad 规则；统一 4 的倍数、奇数 H/W、参考尺度对齐；不可用任意插值掩盖大范围真实错位。
5. **初始化冲突**：旧主干是绝对 RGB，新网络是 `lightup Low + residual`。可选择性迁移 SFE/ResBlock，**禁止照抄旧 RGB tail 和原照明校正头到新初始化**；保存逐模块匹配和未加载清单。训练继续时不要恢复旧模型 optimizer/scheduler state。
6. **梯度检查**：新 Encoder、跨尺度 CSFI、IGAB、IlluminationEstimator、三个参考注入层都应有有效梯度；训练初期记录残差、log_gain、clamp 饱和与 NaN/Inf。
7. **训练/评测一致性**：参考特征和光照估计是否使用全图/Tile 必须一致；Tile 融合后全局照明校正只能执行一次；不使用 GT mean-align、oracle Ref、GT Exposure 统计做推理。
8. **明确数据协议**：不要把官方 LOLv2 Test 100 长期当开发集调参；划分固定、可复现的训练/开发集，最终测试只用于最终评估。
9. **资源验证**：本地只是 CPU smoke tests，必须测真实 GPU 批次显存、step 时间、有效 batch、AMP/GradScaler、梯度裁剪比例。旧参考匹配在较大 patch 上有显存压力。

## 4. 现实数据与预算

| 数据域 | 参考数据约束 | 在项目中的角色 |
| --- | --- | --- |
| LOLv2-Real | 旧记录中训练 `Low∩Normal∩nanobanana_ref_v2` 为 **639** 配对 | 主要源域训练 |
| LOLv2-Syn | 旧记录中训练配对为 **739** | 主要源域训练 |
| LSRW-Huawei | 用户预算约 **500 张 Nano / 设备**（需核对实际可匹配文件数） | **目标域微调** |
| LSRW-Nikon | 同上 | **目标域微调** |

旧资料中曾出现 Huawei/Nikon 各约 250 张、合计约 500 张的版本；**必须以新仓库扫描后的实际 manifest 为准**，不要假设数字已足额。

**预算硬约束**：主要训练必须在 LOLv2-Real 与 LOLv2-Syn 进行；Huawei/Nikon 仅用各自预算允许的 Nano 三元组做参考感知微调。**不得自行把完整 LSRW Low/GT 纳入主训练或目标域训练**；虽然无需增加生成成本，但这仍改变用户指定的数据协议，必须单独获得批准才能启用。

## 5. 训练方案（建议初稿：优先训练完整方法，不先做消融）

### Stage A — 无参考恢复预训练

- 数据：LOLv2-Real + LOLv2-Syn 中的 Low/GT，明确为两域混合采样；`ref=None`。
- 训练：SFE/Encoder/跨尺度 CSFI/ResBlocks/IGAB/IlluminationEstimator/RGB tail；跳过 LTE、SearchTransfer、ref heads。
- 目的：先训练一个有足够恢复能力的主干，避免少量 Nano 配对限制整个恢复网络。
- 启动配方：**80k updates**（建议初稿，非性能保证）；Adam `2e-4`，betas `(0.9,0.999)`，L1；有效 batch 8；Real 256 crop / Syn 128 crop，因单 batch shape 需按域分 batch 或统一尺寸；随机同步几何增强；无参考配对 Mixup 可开启；Cosine 衰减至约 `1e-6`。
- 注意：从头训练或安全 warm start 是可配置策略，不要把旧 RGB tail 直接迁入。

### Stage B — LOLv2 Nano 联合训练

- 数据：Real **639** + Syn **739** 可用三元组（实际数字按清单）。
- 启用完整架构：LTE、SearchTransfer、参考注入、CSFI、IGAB、照明头。
- 建议 **100k updates**（观察独立开发集后调整）；以参考三元组为主，穿插约 **20%–30%** 的无参考 Low/GT batch，防止失去恢复能力。
- 推荐起始 LR 分组（需真实训练微调）：主干/SFE/Encoder `1e-5`；CSFI/IGAB `2e-5`；RGB head `2e-5`；新参考注入 `1e-4`；照明估计 `2e-5`；LTE 初期冻结，必要时小 LR（如 `<=1e-6`）解冻。
- 参考阶段**先不要直接套用普通双图 Mixup**；如果做 Mixup，Low/GT/Ref 必须同规则混合并评估是否仍具有合理匹配语义。
- 保存共享模型；若跨域互扰明显，随后从共享模型分别进行小步数 Real/Syn 域适配（约各 15k 为计划值）。

### Stage C — Huawei 与 Nikon 分别适配

- 从 Stage B 的收敛 checkpoint 开始，使用各自**允许的约 500 个三元组**独立微调；原型目标是每域 20k–40k updates，**需密集验证和 early stopping，不要求强行跑满**。
- 主干 LR 约 `5e-6`–`1e-5`，参考注入/恢复头 `2e-5`–`5e-5`，LTE 首先冻结；固定随机种子与数据清单，避免对小训练集过拟合。
- `Correct Nano / Bypass / Mismatch` 可在开发集作为健康诊断；不需要先安排正式消融矩阵。

### Retinexformer 官方配方中真正核实过的细节

- 官方 LOLv2-Real/Syn 实际 `total_iter=150000`、batch 8、Adam `lr=2e-4`、L1，启用几何增强、`mixup_beta=1.2` 和 gradient clip。
- Real patch **256**；Syn patch **128**；yaml 中渐进 patch 配置是注释，不是生效值。
- `CosineAnnealingRestartCyclicLR(periods=[46000,104000], eta_mins=[3e-4,1e-6])`，注释提到 300k 但生效配置是 150k；实际第一周期因为 `eta_min=3e-4 > base_lr=2e-4` 不属于普通衰减。
- 官方 `image_restoration_model.py` 梯度裁剪实际 `max_norm=0.01`；不要不经梯度统计便套用到新网络。新模型可从 `1.0` 作为监测用启动值，记录裁剪发生率后再调整。
- 这些是**参考配方，不是对新模型的性能保证**。

## 6. 评测与 checkpoint 选择

- 数据范围与单位统一：报告 **RGB PSNR / SSIM**，像素范围 `[0,1]`，`crop_border=0`，图像级求均值，RGB/BGR 顺序固定，避免颜色交换。
- 严禁 GT Mean Alignment、GT as Reference、Test split 上计算均值进行亮度校正。
- 比较时记下 `dataset_manifest_hash`、GT/Low/Ref 文件标识、checkpoint SHA、模型配置、训练图片量、是否使用目标域完整 Low/GT。
- `Retinexformer Huawei ≈ 21.3 dB` 来自**该基线在完整目标域 Low/GT 数据上从头训练的实验**。如果 ASRT 只能用约 500 个 Huawei Nano 三元组做目标域微调，**两者训练数据预算不同，不能将“比不过 21.3”直接解释成架构失败**；可把 21.3 当强参照，但必须单独报告数据协议差异。
- 模型 checkpoint 由固定训练数据中划出的独立 Dev 子集选择，官方 Test 集只用于最终报告；显式记录是否曾反复使用历史 Eval30 作选型。
- 必须同时支持 `full_image` 与 `tiled` inference，并检查纹理查找边界、拼接区域、GlobalIllumHead 只执行一次。

## 7. 建议的新仓库结构

```text
ASRT/
├── README.md
├── LICENSE                         # 核对派生代码许可证
├── pyproject.toml / requirements.txt
├── configs/
│   ├── stage_a_lolv2_restore.yaml
│   ├── stage_b_lolv2_reference.yaml
│   ├── stage_c_huawei_ft.yaml
│   └── stage_c_nikon_ft.yaml
├── asrt/
│   ├── models/
│   │   ├── asrt.py              # 顶层双输入/无参考接口
│   │   ├── csfi_unet.py             # H/4-H/2-H + CSFI + IGAB 主干
│   │   ├── attention.py             # IG_MSA / IGAB / illumination
│   │   ├── reference.py             # LTE + SearchTransfer
│   │   └── legacy_heads.py          # 可选照明校正
│   ├── data/
│   │   ├── paired.py
│   │   ├── triplet.py
│   │   └── manifest.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   └── evaluation/
│       ├── metrics.py
│       ├── inference.py
│       └── diagnostics.py
├── scripts/
│   ├── audit_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── convert_legacy_weights.py
├── tests/
│   ├── test_forward_backward.py
│   ├── test_reference_bypass.py
│   ├── test_triplet_geometry.py
│   ├── test_checkpoint_resume.py
│   └── test_eval_protocol.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   └── PROVENANCE.md
└── .gitignore
```

数据与 checkpoints 一律**仓库外存储**，只提交配置、统计摘要、清单样例，不提交原图、生成参考或大权重；`README` 给出如何挂载数据。

## 8. 新仓库首次里程碑 / Definition of Done

1. 新独立仓库，README/LICENSE/结构与来源说明齐全；旧仓库未被修改。
2. `ASRT(low, ref=None)` 和 `ASRT(low, ref=nano)` 都能前向和反向；无参考分支的所有参考计算/后处理真正跳过。
3. 64 通道 + `8+8+4+2`、H/W 偶数与奇数，原生三个 `T_lv*`，GPU crop 前向、AMP 与梯度完成；关键模块梯度非空、无 NaN。
4. 数据 audit 输出真实 `paired`/`triplet` 规模，严格一一对应，不出现 GT 替代 Ref，不静默减少训练集。
5. A/B/C 三个配置启动运行；至少每阶段一次真实小规模 GPU train+eval+resume smoke，不以 CPU 单张前向冒充完整训练验证。
6. `train.py` 输出 step/LR/loss/梯度/显存/数据来源占比；checkpoint 包含随机状态、optimizer/scheduler、manifest hash、model/config。
7. 固定协议的 Dev 评测，不使用 GT 的亮度对齐；支持单图/Tile，并包含 Reference Bypass 和 Reference Correct 结果。
8. 运行结束交付源码、架构参数量/FLOPs（注明输入尺寸）、训练曲线与 PSNR/SSIM；若没训练完成，明确如实标记阶段，**不得猜测成绩**。

## 9. 后续研究，不阻塞 v1 完整主模型

训练稳定后再考虑 A2：可靠的跨退化结构引导、Correct vs Mismatch Nano、幻觉控制、不同尺度匹配、LoRA 微调及完整消融矩阵。这些不是新仓库第一轮训练的阻塞项。先让完整新架构有竞争力，并把参考贡献的诊断接口预留好。
