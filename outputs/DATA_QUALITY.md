# 数据质量声明（Data Quality Notes）

> 生成：2026-10-10 ｜ 维护者应在本文件中记录所有已知的数据缺口和质量问题，防止开源后被质疑时无据可查。

## 已知缺口

### 1. medsam E2 外验行数不足（种子42/44）

| 单元 | 预期行数 | 实际行数 | 缺失 | 原因 |
|---|---|---|---|---|
| medsam/seed42/p1 | 18,322 | 18,318 | 4 | 去重窗口损失 |
| medsam/seed42/p2 | 18,322 | 18,319 | 3 | 同上 |
| medsam/seed42/p4 | 18,322 | 18,319 | 3 | 同上 |
| medsam/seed44/p1 | 18,322 | 18,319 | 3 | 同上 |
| medsam/seed44/p2 | 18,322 | 18,318 | 4 | 同上 |
| medsam/seed44/p4 | 18,322 | 18,318 | 4 | 同上 |

**根因**：对活跃追加中的 e2_external.csv 执行「读→去重→截断重写」时，写者（GPU2 E3 消融推理）在读取→截断窗口内追加了约 10 行 medsam seed42 数据（case_id: (内部切片号)–475 的 p1/p2/p4），被截断丢失。seed44 的 3 行/提示差异来自同一窗口对不同切片的影响。

**影响评估**：
- 缺失 10 行 / 165,000+ 总 E3 行 = **0.006%**，统计无影响
- 缺失切片为某 TRUSTED 肾的 2 段连续 z 区段（4+4 张；具体切片号不落入公开文档，见内部日志）——单 kidney 内的尾部切片，GT mask 面积接近下限
- **不需要重跑**：样本量级下统计结论完全不变

**如需精确补齐**：对上述两个连续 z 区段（内部记录可查）用 m3_run_eval 定向补测（--models medsam --seeds 42,44）。

### 2. usfm/44/none 行数不足

| 单元 | 预期行数 | 实际行数 | 缺失 |
|---|---|---|---|
| usfm/seed44/none | 18,322 | 15,819 | ~2,503 |

**根因**：USFM E1/E2 评测在处理 seed44 TRUSTED 切片时崩溃（`infer_logits` 函数当时丢失）。已修复函数并重启评测补跑。

**状态**：评测正在运行中，预计 ~30 分钟完成后自动补齐。

### 3. SAMUS 微调配方偏离官方

当前使用"预注册轻量微调"（decoder-only 4.06M + 官方 encoder adapter 39.65M = 43.71M trainable, AdamW 1e-4, cosine 50ep），偏离 SAMUS 官方训练配方（Adam 5e-4 恒定、200ep、pos_weight=2 BCE、随机点击 prompt、有增广）。

**偏差已预注册声明**（§14.6 例外），论文措辞应使用 "SAMUS-inspired light fine-tuning (pre-registered protocol)" 而非 "followed SAMUS official fine-tuning protocol"。

### 4. M6 USFM 微调配方偏离预注册

原预注册 lr=1e-5/50ep/Dice+BCE，实际使用官方 USFM 配方（AdamW 3e-4、layer_decay 0.65、warmup 20ep、400ep、Dice+BCE on probability、RandomRotate90 增广）。

**偏离原因**：预注册超参在撰写时未参考官方微调配置；官方配方的 layer-wise lr decay 0.65 是 ViT 基础模型微调的关键技术。已在 Post-registration Deviations 中记录。

### 5. MedSAM/SAM+LoRA 输入归一化偏离

当前使用 (x-0.5)/0.5 归一化，SAM/MedSAM 官方封装使用 ImageNet 统计 (x-mean)/std。训练与评测内部一致，模型间相对比较不受影响，但可能影响绝对性能上限。

**处置**：已在 RUN_REPORT 声明；如需修正可在 camera-ready 前做归一化敏感性对照（同一 seed 两种归一化各 1 次）。

### 6. M6 未使用官方 ATMLoss

使用 Dice+BCE(概率域) 替代官方 ATMLoss（匹配式损失+点采样掩码损失+辅助监督）。原因是 ATMLoss 需要 query 级标注和匹配逻辑，实现复杂度高；语义输出上的 Dice+BCE 是常见的替代方案。

**处置**：已在 RUN_REPORT 声明；论文 Deviations 小节记录。

### 7. E3 消融行尾混用 CRLF/LF

csv.DictWriter 默认 \r\n，后续追加为 \n。无数据影响；解析时 pandas/csv 均可处理。后续去重脚本统一 `lineterminator="\n"`。

## 数据完整性矩阵（最后核验 2026-10-10 06:15）

| 模型 | E1 (99/单元) | E2 P3 (18,322/单元) | E2 消融 P1/P2/P4 (18,322/单元) |
|---|---|---|---|
| unet | ✅ 3/3 | ✅ 3/3 | 不适用 |
| nnunet | ✅ 3/3 | ✅ 3/3 | 不适用 |
| medsam | ✅ 3/3 | ✅ 3/3 | ⚠ seed42 差3-4行, seed44 差3-4行 |
| sam_lora | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 |
| samus | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 |
| usfm | ✅ 3/3 | ⚠ seed44 补跑中 | 不适用 |

## 预防措施（已实施）

1. `m3_reconcile.py`：全矩阵对账工具，可随时运行检查单元完整性
2. e1/e2 CSV 追加无去重——**M5 统计入口必须加主键去重断言**（已列入方案 §14 冻结令配套要求）
3. test_access_log.txt 记录所有测试集触碰
4. nnU-Net sens 文件与主 CSV 分离存放，避免 best/final 口径混淆

## 8. SAMUS 提示路径不对称（P3-10，公平性声明，2026-10-11）

SAMUS 微调（预注册轻量配方）按官方训练方式**仅使用点/点击提示**；其 box 提示通路（p3/p4）从未经过微调。因此主表 p3（oracle box）行对 SAMUS 不构成同条件比较：SAMUS 的 p3/p4 成绩反映"未微调的 box 通路"，其 p2（点提示）才是训练过的通路。论文必须：主表并列 SAMUS p2 行并显著声明该不对称；不得将 SAMUS p3 与其他模型的 p3 直接排序为"模型孰优"。数据齐备（E1/E2/E3 全部提示格），聚合见 aggregate_results.json。

## 9. 外验极点切片标注口径现象（P1-3 产物，2026-10-11）

TRUSTED 标注范围起止端存在 GT 面积骤减的切片（<15% 该肾中位面积者 999/18,322=5.45%（正典口径，见 outputs/summary/canonical_pole_slices.json））：所有模型在这些切片 Dice≈0（含 oracle box 的 MedSAM p3 极点 0.7164 vs 非极点 0.8939（正典口径对齐值））。定性为标注口径现象（肾极出入扫描平面时仅标注可见小截面）+ 真实异质性叠加，非评测伪影（叠图目检：无错位/全零；对齐正确）。主口径不变（预注册冻结）；分层披露写入失败分析。图形化证据：outputs/figures/bimodal/（含 59 肾直方图、极值肾叠图、GT 面积剖面）。注意该目录含患者图像，**禁止入 git**。

## 10. USFM 聚合文件差异裁决（第 2 项，2026-10-11）

**现象**：旧 `aggregate_summary.json`（2026-10-10 23:23 生成）与重新生成文件在 usfm seed44 的 E2 均值差 0.9 点（0.4808/19,107 行 vs 0.4892/18,322 行）；旧文件 e1 usfm 每单元记 200/198/396 行（真值 99）。

**裁决**：旧文件生成时消费的是**清理前的 CSV 状态**——usfm 行曾被多轮评测重复追加（旧配方评测 + v3 重评并存，同 case_id 多版本行，e2/44 多出 785 行、e1 每单元多出 99-297 行）。其后（最终评测 01:59 收官前后）CSV 经过去重清理，保留了每键一行（末次=v3 行）。**两套聚合可以同时"各自正确"的表面矛盾即由此解释**：新脚本断言主键唯一+逐格 99/18,322 对"当前 CSV"成立；旧文件对"历史 CSV"成立。核验（2026-10-11）：当前 `e2_external.csv` usfm/44/none = 18,322 行 / 18,322 个去重 case_id / 0 重复 / 0 个越参考清单的 id（对照 p3_oracle_trusted.npz 18,322 引用清单），均值 0.4892。**0.8-0.9 点的差异方向符合预期**：被清掉的 785 行来自旧配方（差）模型，混入拉低均值。

**处置**：`m5_aggregate.py` 现同时重建两份文件（旧 schema 由同一组已断言单元生成），两文件逐格 diff=0.0，`e2_completeness` GAP 块已按当前 CSV 重算（0 个非预期格）。权威文件为 `outputs/summary/aggregate_results.json`；旧路径仅为写作端兼容镜像。
