# 给 Codex / 代码助手的一次性执行指令：建立 ASRT 新仓库

你是一名负责 PyTorch 低光图像增强研究工程的代码代理。请阅读同目录 `ASRT_PROJECT_BRIEF.md` 作为**唯一需求基准**，然后实施新项目。

## 任务目标

**建立独立新仓库 `ASRT`**，实现 `Illumination-Guided Cross-Scale Reference Restoration for Low-Light Image Enhancement`。不要直接在旧 `TTSR-lowlight` 仓库继续叠补丁，也不要先启动结构消融。旧项目仅是参考来源，完整新架构是第一目标。

可用旧原型：`ttsr_csfi_unet_v3.zip`（包含 `MainNetCSFIUNet.py`、7 个 CPU smoke tests、安装与 warmstart 脚本）。可以借鉴而不必完全照抄；请首先审计其已知缺陷并修复。

## 不可更改的架构约束

- 双输入 `low, ref`；允许 `ref=None`，严禁训练时用 GT/HR 代替缺失 Ref。
- Low U-Net Encoder：`E0@H, E1@H/2, E2@H/4`；默认通道 64。
- 三级跨尺度 TTSR-style 流：Coarse@H/4 输入原生 `T_lv3`；Middle@H/2 输入原生 `T_lv2`；Fine@H 输入原生 `T_lv1`，分别有 U-Net Skip。
- 两尺度双向 CSFI2、三尺度互联 CSFI3。H/4、H/2 各 1 个 illumination-guided IGAB；IG-MSA 是 channel-transposed attention，不是 HW×HW attention。
- 默认深度 `8+8+4+2`，相当于 SFE 8、粗流 8、中流 2×4、细流 3×2 个 ResBlocks。
- `IlluminationEstimator` 预测照明特征和 log_gain；RGB 输出为提亮输入基础图加主干 3 通道残差。`GlobalIllumHead`/`RefIllumTransfer` 可选且前后处理协议一致。
- VGG/LTE/SearchTransfer 的参考特征值路径保持来自 **真实生成参考**，不能使用 GT 特征作为部署时的参考值。

## 训练需求与数据硬约束

- **主训练仅使用 LOLv2-Real 与 LOLv2-Syn**，目标域 Huawei/Nikon 只允许用真实可用的约 500 Nano 三元组/设备微调；任何超出目标域数据预算的训练需先征得用户同意。
- Stage A: 无参考 Low/GT 恢复预训练，约 80k steps（启动值）；L1、Adam `2e-4`、effective batch=8、Cosine；Real 256 crop，Syn 128 crop，记录混合采样策略。`model(low,None)` 必须真实可用。
- Stage B: LOLv2 Real/Syn Nano 联合训练，约 100k steps（启动值），约 20–30% 无参考 batch；参考分支更高 LR，已学主干更低 LR；同步几何增强。
- Stage C: Huawei/Nikon **分别**从 Stage B checkpoint 以其可用三元组短周期适配，约 20k–40k steps 为计划值，密集独立 Dev 评测及早停。不要重复过采样到没有意义的 100k+ steps。
- 官方 Retinexformer 配方是学习来源，而非必须机械复制：LOL v2 官方有效 `150k` steps，Real 256 crop / Syn 128 crop，batch 8，Adam 2e-4，L1，Mixup，CosineAnnealingRestartCyclicLR；实际 code clip `0.01`，不要无监测照搬。

## 执行顺序

1. 在有权限的情况下创建 `ASRT` 仓库或在独立目录初始化 Git 仓库。若没有权限，就完整建立本地项目并说明阻塞，不能假称已推送。
2. 从旧仓库和 v3 原型抽离可重用代码，建设独立 package、训练器、数据集、评测器、文档与 4 个 YAML。
3. 第一优先修复：`ref=None` 的完整前后向；参考 mask/diagnostic 在 S=None 时不崩溃；输入 Pad/Unpad；参考数据严格存在性校验；参考注入初始值不破坏无参考主干；推理的整图/Tile 一致性。
4. 增加 unit/integration tests：odd size、三个 native reference scale、gradient path、三元组同步几何增强、no-ref、torch amp、checkpoint resume 和 PSNR/RGB 数据范围。
5. 完成代码静态检查与所有 CPU smoke，随后在配置的实际 GPU 环境完成小批量数据真实训练/评测/断点续训 smoke；记录显存、step time、数据统计、梯度范数。
6. 实施 Stage A→Stage B→Stage C 训练脚本与配置（无需一口气在本次对话内完成上百 k steps，除非用户明确要求实际开跑）；不先搭建消融矩阵。
7. 交付新项目的路径/URL、目录树、可复现启动命令、测试报告、初始化/权重迁移说明、数据 manifest 与已知限制。任何没跑的训练、没验证的性能必须写清楚。

## 参考源码与许可

- Legacy：`https://github.com/apeanut60/TTSR-lowlight`，分支 `experiment/ttsr-lowlight-lolv2real`
- Original TTSR：`https://github.com/researchmm/TTSR/blob/master/model/MainNet.py`
- Retinexformer：`https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/models/archs/RetinexFormer_arch.py`
- 官方 YAML：`https://github.com/caiyuanhao1998/Retinexformer/tree/master/Options`

**不要凭空发明新结构、不复现论文中不存在的损失、不伪造新基线成绩。先实现我们已经敲定的完整模型。**
