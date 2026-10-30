# ASRT-v3 Reliability Diagnostics

训练日志 `train.jsonl` 在有参考 step 记录：

| 字段 | 含义 |
|------|------|
| `c_t` / `c_l` | mean / std / q10 / q50 / q90 / min / max |
| `inject_res_norm` | coarse/middle/fine 纹理残差 L2 |
| `teacher_l1` | 无参考 batch 上相对 Stage-A Teacher 的 L1 |
| `stable` | 是否仍在稳定期 |
| `lr_groups` | 各 param group 当前 LR |

运行时 `model.last_aux` / `backbone.last_v3_diag` 另含：

- `c_t`, `c_l`, `c_t_fine` 空间图
- `delta_l_coarse` / `delta_l_middle`
- `match_diag`（index / score / margin / cycle / spatial）

## 健康检查

| 症状 | 含义 |
|------|------|
| `C_T` 全≈0 或全≈1 | 可靠性退化；检查 head 初始化与 LR |
| `inject_res_norm` 全≈0 且 `C_T` 高 | Head / Gate 异常 |
| `delta_l_*` 范数爆炸 | Adapter 失控；确认零初始化与 `d_ref` clamp |
| `teacher_l1` 持续上升 | 无参考能力漂移；可上调 `teacher_consistency_weight` |

## 可视化建议（评测脚本）

对 Dev/Test 样例导出：`C_T`、`C_L`、match 位置着色、注入残差热力、`P_ref_matched`。  
对比暗平坦区 vs 结构区的平均 `C_T`（预期：结构区更高）。

**禁止**用官方 Test 调门控阈值；诊断仅服务 Dev 与训练监控。
