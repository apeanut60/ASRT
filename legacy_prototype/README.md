# TTSR-lowlight v3 · CSFI-U-Net + Retinexformer IGAB

这是为 `apeanut60/TTSR-lowlight` 的 `experiment/ttsr-lowlight-lolv2real` 分支准备的**可安装实现包**。它不会覆盖或修改旧版 `MainNetEnhance.py`，也不会修改 `LTE`、`SearchTransfer` 或训练集。建议在本地新分支上安装。

## 整体计算图

```text
Low [-1,1] ──┬──────────> IlluminationEstimator ─> illum_h4 / illum_h2
             │                                         │
             v                                         │
            SFE (H, 64) ────────────── E0 skip         │
             v                                         │
           Down (H/2,64) ───────────── E1 skip          │
             v                                         │
           Down (H/4,64)                                │
             + T_lv3 (256 ch), S_h4                    │
             v                                         v
           RB11 + IGAB H/4 <───────────────────── illum_h4
             |\
             | \── Up + E1 skip + T_lv2 (128 ch), S_h2
             |                          |
             +────── CSFI2 <───────────+
             |                          |
           RB21                       RB22 + IGAB H/2 <─ illum_h2
             |                          | \
             |                          |  \── Up + E0 skip + T_lv1 (64ch), S_H
             |                          |                       |
             +──────────────────────────+───────── CSFI3 ──────+
             |                          |                       |
           RB31                       RB32                     RB33
             \__________________________|______________________/
                                        v
                          Multi-scale Merge + RGB residual
                                        +
                    Low -> estimated positive light-up gain
                                        v
                       optional old RefIllum + GlobalIllum
                                        v
                                    enhanced RGB
```

**关键约束**：三个 stream 现在是真正的 H/4 / H/2 / H；SearchTransfer 的 `T_lv*` 保持原生尺度；原 MainNet 的阶段名称和 ResBlock 参数形状尽量保留；在 H/4 和 H/2 插入 IGAB（通道注意力 + DWConv FFN）。

## 安装（在你自己的 Git 仓库中执行）

解压本包后，在包目录执行：

```bash
cd /path/to/TTSR-lowlight
git switch experiment/ttsr-lowlight-lolv2real
git switch -c experiment/v3-csfi-unet

python /path/to/ttsr_csfi_unet_dropin/scripts/install_csfi_unet.py \
    --repo-dir "$PWD"
```

安装器自动复制 `model/MainNetCSFIUNet.py`，修改现有的 `model/TTSREnhance.py`，并在 `option.py` 添加：

```text
--mainnet_arch legacy|csfi_unet
```

原架构默认依然是 `legacy`；指定 `--mainnet_arch csfi_unet` 启用新结构。修改前会备份原源文件到 `*.before_csfi_unet.bak`。请用 `git diff` 审核安装器引起的改动。

## 旧权重安全迁移

旧模型是绝对 RGB 回归，新模型是 `light-up Low + RGB residual`；**不要直接加载旧 `merge_tail` / `RefIllum` / `GlobalIllum` 权重**，否则容易出现很大的初始化偏差。使用迁移脚本：

```bash
python /path/to/ttsr_csfi_unet_dropin/scripts/warmstart_csfi_unet.py \
    --input /path/to/old_checkpoint.pt \
    --output /path/to/v3_warmstart.pt \
    --num_res_blocks '8+8+4+2' \
    --n_feats 64
```

该脚本把旧模型的 SFE 和 ResBlock 等**同形状参数**复制到新 MainNet，保留参考 LTE 等非 MainNet 权重；跨尺度 CSFI、Encoder/Skip、IGAB、照明估计、RGB 残差头从 v3 初始化。为稳健起见默认不复制参考注入头（不同尺度下输入统计改变）；如要尝试可加 `--copy_injection_heads`，但不推荐第一轮使用。

迁移输出是一个包括 `MainNet.*` 的 state_dict，可走旧模型已有的权重加载接口。注意检查旧训练入口的 `load_pretrain` 标志如何传递，以及加载日志中的 `matched / missing / skipped` 数值。`--pretrain_require_core False` 能放宽旧 checkpoint 的严格加载检查，但更推荐使用上面的完整迁移 checkpoint。

## 原训练命令如何接入

沿用现有 LOLv2 Nano 数据集和训练脚本，只需要在原训练命令末尾追加（前提是该脚本把 options 传给 `TTSREnhance(args)`）：

```text
--mainnet_arch csfi_unet
```

如果使用了 checkpoint 初始化，应指向 `v3_warmstart.pt` 并确认新模块可训练；不要把旧的 optimizer 状态恢复进新模型。

**不需要先做消融。** 直接训练完整 v3：三尺度 Encoder/CSFI、原三级 reference matching、coarse+middle IGAB、照明估计和残差重建同时启用。等主模型训练正常后，再单独做各模块消融。

## 自检

```bash
cd /path/to/ttsr_csfi_unet_dropin
python -m unittest discover -s tests -p 'test_*.py' -v
```

已覆盖：
- H/W 为 4 的倍数和非 4 的倍数；
- 三条原生尺寸参考 `T_lv3/T_lv2/T_lv1`；
- 旧的 `gate_mode`、`stage_mode_*` 关键字参数；
- 参考特征对输出有实质影响；
- IGAB 的多头维度；
- 主要模块和参考特征的反向传播梯度；
- 安装脚本可重复执行；
- 迁移权重不会错误复制绝对 RGB 输出头。

## 已知限制 / 尚待在服务器验证

1. 当前本地只有 PyTorch CPU 环境；未运行服务器上的真实数据集训练，也未验证新架构的 PSNR、GPU 显存或训练稳定性。形状/梯度测试通过不等于指标提升。
2. 老版本专门读取 `last_gate_stats` 或 `last_residual_decomp` 的**详细诊断脚本需要适配**：新网络保留属性，但不伪造旧结构的统计数据；当前返回 `None`。
3. 当前 `TTSREnhance` 已有 VGG/SearchTransfer 路径是全模型的计算和显存开销的重要来源，新架构增加编码器和 IGAB 也会增加负担；须以真实 crop / batch 测量。
4. `GlobalIllumHead` 与 `RefIllumTransfer` 延续旧后处理接口；在旧训练器中 tile 推理会先对 tile 恢复，再对整图应用两者。为了可比性，训练和评测应核对它们的启用标志。
5. 由于旧 TTSR 的 `num_res_blocks='8+8+4+2'` 是深度配置，第一轮可完全沿用；不要在不记录的情况下另改深度、loss、数据配比。
6. 改变过 mainnet 的 `state_dict` / API / train script 的其他分支，需要先 review 自动安装结果，必要时手工调整。
7. 只保证结构设计可测试，不保证论文新颖性：真正的研究贡献仍是后续 A2 可信参考结构引导，而非直接组合 U-Net、CSFI、IGAB。

## 源码参考

- TTSR original MainNet: https://github.com/researchmm/TTSR/blob/master/model/MainNet.py
- Retinexformer IG-MSA/IGAB: https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/models/archs/RetinexFormer_arch.py
- 当前 MainNetEnhance: https://github.com/apeanut60/TTSR-lowlight/blob/experiment/ttsr-lowlight-lolv2real/model/MainNetEnhance.py
