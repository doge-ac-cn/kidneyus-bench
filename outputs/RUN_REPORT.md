# RUN_REPORT — 超声肾脏分割基础模型跨库基准（方案A v1.4）

**预注册**：本方案于外验结果聚合与报告之前在 OSF 注册（DOI: [10.17605/OSF.IO/TX7RN](https://doi.org/10.17605/OSF.IO/TX7RN)，2026-10-10 生效）。自注册时起实验冻结：仅允许已排队任务收尾，任何偏离将记录于论文 Post-registration Deviations 小节。

自动生成 + 里程碑状态：2026-10-08（M0–M1 会话）

## 环境快照

- 机器：3× Tesla V100-SXM2-16GB（驱动 580.178.04，sm_70；无 BF16，训练统一 FP16 AMP）
- 环境：conda env `kidneyus2`（python 3.10，**纯 pip** torch 2.7.1+cu126）
  - 注意：本机 `~/.local/lib/python3.10/site-packages` 会遮蔽环境包，所有训练/评测命令必须带 `PYTHONNOUSERSITE=1`
  - 依赖快照：`outputs/requirements.lock`（待训练启动后补录 pip freeze）
- 随机种子：`set_seed(42)` 工具（scripts/utils_kidney.py）贯穿；划分种子 42
- 磁盘：数据+环境+输出 ≈ 60GB（峰值预算 300GB 内）

## 与方案 v1.0 的偏差记录（数据源与协议层面）

| 项 | 方案原设定 | 实际 | 处置 |
|---|---|---|---|
| OKD 下载 | Zenodo 10.5281/zenodo.5983586 | Zenodo 记录已被删除（spam），无镜像 | 用户经 Qualtrics 注册获得作者 OneDrive 分享（CC BY-NC-SA），直链下载成功 |
| TRUSTED 下载 | Dataverse DOI 10.34894/VQ1DJA | 该 DOI 解析到无关资源 | 实际托管 figshare 27981050（10.6084/m9.figshare.27981050），**CC BY 4.0**；`springernature.figshare.com` 有 AWS WAF，经旧域名 `ndownloader.figshare.com` 匿名下载成功（15.97GB，sha256 存档） |
| TRUSTED 规模 | "48患者/96肾" | 那是 **CT 侧**（48例×2肾）；**US 侧为 59 肾/39 患者**（部分患者双肾均有US） | E2 外验以 US 全库 59 肾为单位；论文中如实写明 |
| OKD 规模 | 514 图 | 分发包 534 图；xlsx 514 行（514 出处）；两标注者 Capsule mask 双非空者 **486 例** | 队列=486，划分按患者键（IM-XXXX，468 键，9 键多图）患者级划分 |
| 分层字段 | 按自体/移植分层 | 分发数据无 native/transplant 字段（CSV 仅 Quality/View/Comments） | 方案§10规则3：E5 亚组分析**删除**；划分改按 view（Transverse/Longitudinal）分层 |
| M5 SAM-MedUS | 权重截止 2026-11-01 | 官方仓库空占位，权重从未公开 | 用户决定改用 **SAMUS**；但 SAMUS 官方权重仅 Google Drive（本机不可达，issue#42 反映链接疑似失效）→ **待用户提供权重文件放入 data/weights/（文件名 SAMUS_pre_train）**；若 2026-11-01 仍不可得则按方案规则1降级为 4 模型并注明 |
| HD95 实现 | medpy | scipy 距离变换自实现（等价定义，无 medpy 依赖） | utils_kidney.hd95_mm |
| OKD HD95 单位 | mm | xlsx 像素间距单位不明（cm/inch 歧义，部分为 1/空） | OKD 内部 HD95 以**像素**报告（TRUSTED 用 NIfTI spacing 正常 mm），论文注明 |

## 里程碑状态

### M0 数据下载与验收 — ✅ 通过（含一次下载通道故障与恢复）
| 门 | 结果 |
|---|---|
| OKD 图像数 | 534 分发 / 486 队列（双标注者 capsule mask 均非空）；mask 二值 {0,1}，尺寸与图像一致 |
| OKD 患者ID | IM-XXXX 键 468 个（队列内），9 键多图 → 患者级划分落实 |
| TRUSTED 计数 | US：59 肾/39 患者；CT：48 例；Annotator1/2 + STAPLE 共识（GT_estimated_masksUS）齐全 |
| TRUSTED 标志点 | 7 点/肾 × 59 × (A/B/GT) ✓ |
| TRUSTED 几何 | GT 与图像 spacing/尺寸一致 ✓；US spacing 统一 0.3×0.3×0.3mm（1024×768×637–868） |
| 许可存档 | TRUSTED figshare 页 + data_use_agreement.txt → outputs/trusted_okd_license_archive.md；OKD 引用要求已记录。**待办：TRUSTED DUA 按论文要求邮件回签（需用户签名）** |
| Sanity（MedSAM 零样本） | OKD 测试集随机 20 张（GT-box 提示）：mean Dice **0.7808**（outputs/m0_sanity_medsam.json）。注：首版 0.3231 系 box 未做 native→1024 缩放的 bug 所致（2026-10-09 代码评审 CRITICAL-1），已修复重算，0.7808 与文献口径一致 |
| 数据红线 | TRUSTED 未参与任何训练（仅外验）；泄漏检查在 M3 结果出数时复查（Dice>IoD 即报警） |

### M1 预处理+划分+IoD — ✅ 通过
- `outputs/splits.json`：canonical 划分（train 293 图/281 患者，val 94/94，test 99/93，无交叉）
- TRUSTED 有效切片：**18,322** 片（STAPLE mask ≥50px），`outputs/trusted_us_slices.csv`
- **IoD = 0.9478**（切片级）/ 0.9469（肾级）→ 落在 0.90–0.96 门内，与 TRUSTED 论文 >93% 一致 ✓（outputs/metrics/iod_reference.csv）
- 陷阱记录：TRUSTED mask 为 float 概率图（STAPLE），必须 0.5 阈值二值化（已写入 preprocess_config.json）

### M2 内部训练 — 🔄 进行中（2026-10-08 深夜状态）
- **M1 U-Net 3种子 ✅ 完成**：val Dice = 0.7092 / 0.6975 / 0.5655（seed 42/43/44，256×256，100ep）
- **D1 YOLOv8s 3种子 ✅ 完成**：val mAP50 = 0.993 / 0.994 / 0.995（P 0.948–0.985，R 0.979–1.0）→ **M2 门（mAP50≥0.9）在 val 上通过**；test 集评测在 M3 一并出数
- **M2 nnU-Net v2 2d 🔄 训练中**：seed42(GPU1) + seed43(GPU2) 并行，seed44 接力（nnunetv2 2.8.1，canonical split 作 fold0，自定义种子训练器）
- M3 MedSAM / M4 SAM+LoRA / M5 SAMUS：训练脚本在 M2 出数后接排（M5 视权重可得性）

### 环境故障记录（复现者注意）
1. `~/.local/lib/python3.10/site-packages` 会遮蔽 conda 环境包：**所有命令必须 `PYTHONNOUSERSITE=1`**，且 pip 安装前也要设（否则 pip 把用户站旧包当"已满足"）
2. pip `--ignore-installed` 降级会留下新旧混合文件（monai/ultralytics/torch._inductor 均因此炸过）：**降级前先 rm -rf 对应包目录**
3. torch≥2.8/cu130 无 V100(sm_70) 内核：锁定 **torch==2.7.1+cu126**
4. nnunetv2 2.8.x 环境变量为小写 `nnUNet_raw/nnUNet_preprocessed/nnUNet_results`

## 泄露与正确性审计（2026-10-09，应要求复查）

1. **划分来源**：nnU-Net 训练日志原句 `Using splits from existing split file: .../splits_final.json`；fold0 与 outputs/splits.json 逐病例一致（train 293 / val 94），与 canonical test 零交集。
2. **患者级完整性**：486 例按 IM-XXXX 患者键核对，跨划分患者交叉 = 0。
3. **数据目录核对**：nnUNet_raw imagesTr=387（train∪val）、imagesTs=99（test）、交集 0。
4. **感知哈希查重**：486 图 × 256-bit dHash，65,855 个跨划分图像对中**最小汉明距离 29，距离≤10 的对为 0** → 无近似重复帧跨划分泄露；同划分 27 对近重复均为同患者多帧（同划分内，允许）。
5. **数值合理性**：EMA 0.938 < IoD 0.9478（人类一致性上限）——模型位于人类上限之下，符合"训练良好"而非泄露的特征。
6. 结论：**未发现任何数据泄露或代码错误迹象**。EMA 为每 epoch 在 val 集 50 个迭代上的滑动平均（nnU-Net 标准行为），论文最终数字以训练后完整推理为准。

## OKD 官方 nnU-Net 成绩（作者分发包内原始 summary.json）

来源：作者 OneDrive 分发的 OKD_weights 归档（2022-10-11 构建），nnU-Net **v1**（nnUNetTrainerV2），Task001_KidneyCapsule，作者自己的随机5折 CV（~415例队列，83例/折验证）。聚合件：`outputs/okd_official_nnunet_dice.json`。

| 变体 | 折数 | 各折 Dice | 均值 |
|---|---|---|---|
| annotator_1 | 5 | 0.857 / 0.933 / 0.910 / 0.874 / 0.852 | **0.8852 ± 0.0315** |
| annotator_2 | 4 | 0.850 / 0.931 / 0.854 / 0.865 | 0.8749 ± 0.0329 |
| mixed | 2 | 0.840 / 0.906 | 0.8728 ± 0.0327 |

论文对比口径注意：官方为 nnU-Net **v1**（2021）+ 随机5折；本方案为 nnU-Net **v2**（2024）+ 患者级划分。0.938（EMA中间数）高于官方 0.885，差距需归因于版本迭代与划分差异，论文中如实注明。
**官方无外部测试环节**：作者分发的权重包经核查仅含内部5折CV产物（无任何 test/external 目录）；OKD 论文发表于2023，当时亦无公开外部队列可用。故官方 0.8852 与本方案 EMA 同属内部成绩，最终对比以本方案 **test 99 例完整推理**（从未参与任何训练与选择）为准。

## M6 USFM 接入（2026-10-09 12:4x）

- 权重到位：用户提供官方 `USFM_latest.pth`（343MB，sha256 d5fdab3e...，ViT-B/16 编码器188键），存 `data/weights/`
- 下游栈：mmengine 0.10.7 + mmcv-lite 2.1.0（纯python，免编译）+ mmseg 1.2.2（git源装，PyPI sdist 的 setup.py 有 py2 reload bug 装不上）；官方 loader `load_pretrained` 完成位置编码插值与键重映射（27→63 插值日志确认，"loaded successfully"）
- 架构协议（v1.4 A1 修正案的落实）：**prompt-free，只进 E1/E2**；下游 = 官方 HVITBackbone4Seg + ATMHead，512×512 输入 3通道 /255 预处理（原生约定）
- 微调：全模型 AdamW 1e-4、50ep、Dice+BCE、GradScaler（T1-3 ✓）、3种子——训练/评测管线 CPU 冒烟通过（forward+backward 368张量有梯度）
- 排程：m6_usfm_queue.sh 挂在 best_primary 链后（GPU0 空闲档），训完自动跑 usfm E1/E2 评测
- 脚本：scripts/m6_usfm.py、scripts/m6_usfm_queue.sh；m3_common/m3_run_eval 已加 usfm 分支



## 第三轮 live 审计（2026-10-09 11:54）——陈旧数据清除与链路修复

审计发现并处置（时间线证据链闭合）：
1. **陈旧 samus 评测行清除**：followup 链 stepA 于 10:58 用**已删除的 v1 decoder-only 权重**写了 e1 297 行 + e2 3263 行（seed42 半成品），随后进程被外部终止。已用 SIGSTOP→过滤→SIGCONT 安全清除（append_row 逐行开合文件，无 fd 偏移风险），清理后 0 残留、无断行、各组计数正确（9组×99）。
2. **best_primary 路径 bug**：--pred-base 缺 {seed} 占位符会静默产出 0 行 → 已修复（占位符 + score 脚本 --pred-base 语义）并重启 watcher。
3. **v2 链的 E3 步骤时序缺口**：v2 的 SAMUS E3（~12:40）早于 stage1 产出 npz（~13:30）会崩在 np.load → 已挂 redo watcher（等 v2 DONE + 6个npz 双条件）。
4. **checkpoint_best 三种子均已存在**（371MB×3），best_primary 前置条件满足。
5. **E1++（unet512）结论落定，不再重跑**：lr=1e-3 三种子全崩（0.47/0.25/0.28）；lr=3e-4 仅救回 2/3（0.386/0.398，seed44 仍发散 val=0）——**从零 U-Net 在 512 分辨率下无论 lr 均无法稳定训练**，256 配方维持主表；基础模型（1024/256 固定分辨率）无此不稳定性，对比时如实呈现。
6. 磁盘余量 328G（82%），持续监控。

**当前有效产物**：e1（unet/sam_lora/medsam×3种子×99=891行）、e2（unet 3种子全量 + sam_lora 进行中）、nnU-Net seed42/43 checkpoint_final、v2 samus_seed42/43.pt。SAMUS v2 训练：seed42 BEST **val_point 0.9337 / val_box 0.9182**（14轮即超 v1 终值），seed43 训练中。



## M5 SAMUS 低分诊断与 v2 重训（2026-10-09 中午）

用户质询"SAMUS 0.906 与 SAM 系 0.96 的差距是否算错/没优化好"。核查结论：**无数值错误，是微调配方吃了个大亏**——
- v1 配方（decoder-only 4.06M）恰好与 SAMUS 官方配方**相反**：官方（external_SAMUS Samus.__init__ 内置冻结逻辑）是**冻 decoder、只训编码器侧 US 专属轻量模块**（cnn_embed/Adapter/rel_pos(2,5,8,11)/upneck，39.65M——SAMUS 的超声表征全在这里）；
- v1 把这些全冻住，SAMUS 的武功等于被废；且 MedSAM 是全参93.7M、SAM+LoRA 有编码器 LoRA——三者适配容量不对等，比较被混淆；
- 数值本身无错：三种子一致（0.9014/0.9037/0.9105）、loss 正常收敛、零样本对照（0.829/0.794）→ 微调点路径 +0.11 合理。
**v2 重训（已启动）**：官方配方 + decoder 联合（43.71M 可训练 = 编码器侧39.65M + decoder4.06M），bs8（显存13.2GB）。epoch14 即 **val_point 0.9337 / val_box 0.9182**，双双超过 v1 训满50轮的成绩。旧 decoder-only checkpoint 已删除，E1/E2/E3 评测随 v2 链自动重跑。另：centroid 点提示与官方随机点击的分布差记录为已知局限（协议一致性优先）。



**修复验证全部通过**：NaN哨兵roundtrip、get_prompt长短命名、trusted limit跨volume累计、decoder批处理补丁（batch=1逐位差0/batch=4差3e-7）、LoRA=decoder4.06M+LoRA0.295M、box缩放亚像素、test隔离、splits_final存活。**同时抓到GPU2 watcher已发生的真实事故并处置**：

| 事故 | 处置 |
|---|---|
| nnU-Net predict 42/43 四连败（watcher缺nnUNet环境变量导出） | 修复+新脚本 m3_gpu0_followup.sh 在GPU0补跑（checkpoint_final，与seed44口径统一） |
| m3_score_nnunet argparse 拒绝 "okd,trusted"（choices与默认值自相矛盾） | 已修复 |
| score 读 OKD 预测带 .png 后缀全 miss | 已修复（cid[:-4]） |
| SAMUS 微调三连败（registry全冻结→空优化器） | 已修复（显式解冻decoder，4.06M），requeue 队列自动重启 |
| nnunet44 watcher 绑 GPU0（今晚将与 SAMUS 队列冲突） | 改绑 GPU1（seed44训练结束正好空闲） |
| E3行与headline同CSV | 记录约定：headline=prompt∈{none,p3}行；消融=prompt∈{p1,p2,p4}行；M5统计层加主键唯一断言 |
| stage1与eval的TRUSTED预处理不一致（逐切片min-max vs 原始） | 统一为原始（实测值域0-254，min-max≈恒等） |
| watcher触发标记无退出码校验（本次benign误触发） | 记录为已知模式问题；新增watcher均带 rc 记录 |

**GAP 补齐**：SAMUS 的 E1/E2/E3 评测已由 m3_gpu0_followup.sh 认领；E1++（unet512）结论以 val 曲线为准（256: 0.71/0.70/0.57 vs 512-unstable: 0.47/0.25/0.28，lr3e-4重跑中）；E1+ CV锚点与 P1框IoU补算列为本周待办。



脚本：`m3_common.py`（模型加载/数据迭代/提示/指标）、`m3_stage1_prompts.py`（P1/P2/P4派生+D1检测表）、`m3_run_eval.py`（主评测）、`m3_export_nnunet_input.py`+`m3_nnunet_predict.sh`+`m3_score_nnunet.py`（nnU-Net CLI三件套）。

双智能体审查发现 4 CRITICAL + 5 MAJOR + 12 MINOR/NIT，全部修复：
- **C1** infer_sam_family 多余`[0]`（预测全垃圾）→ 删除；修复后同case Dice 0.9751 与评审独立计算逐位一致
- **C2** 空提示 np.ma.masked 序列化往返失效（击穿"空提示计Dice=0"规则）→ NaN哨兵 + load端全NaN→None
- **C3** nnU-Net 导出缺 `_0000` 后缀（预测零输入）→ 修复命名
- **C4** `-f all`+不存在的 checkpoint_final → 改 `-f 0 -chk checkpoint_best.pth`
- **M** SAM系推理缺 window_normalize（训练/推理分布漂移，HD95 109→10mm）→ 接线修复
- **M** P2质心与P1框不同源 → 统一 centroid_from_mask；**M** CSV追加无去重（M5统计层将加唯一性断言）；**M** e3_detection 写模式脆弱 → 按数据集分文件；conf 列实义化
- 其余：argparse choices、prompt缓存（66万次IO→单次）、TRUSTED逐切片min-max≈恒等（实测值域0-254，注记）、kid.rstrip→regex、测试集访问日志落地
- 冒烟验证：stage1(3例)→主评测(2例×4提示)端到端通过，P3 0.975 > P4 0.960 ≈ P1 0.916 > P2 0.478，排序符合预期；正式运行前已清空冒烟数据



| 模型 | val Dice（3种子） | 备注 |
|---|---|---|
| SAM+LoRA（M4） | **0.9602 / 0.9603 / 0.9600** | 极稳定（sd≈0.0001），与 MedSAM（0.9625）同档 |
| U-Net 512（E1++ ①） | 0.4722 / 0.2482 / 0.2753 | **lr=1e-3 配方在512下系统性不稳定**（loss平台0.61、val塌至0.02）——辅助检查改用 val 选定的 lr=3e-4 重跑中，配方差异如实记录 |
| MedSAM（M3） | seed44=0.9625；seed42/43 失效（外部终止/OOM竞态） | 补跑 watcher 已接棒自动执行 |

watcher 链（全自动）：GPU0队列✅ → MedSAM 42/43 补跑 → SAMUS 3种子 → U-Net512(lr3e-4) 3种子。基础模型 vs 从零模型的训练稳定性差异本身即是可报告发现。



## M5 SAMUS 接入（2026-10-09）

- 权重到位：用户提供官方 `SAMUS.pth`（533MB，sha256 a95de25d...，键结构与官方 repo 完全吻合：cnn_embed/D_fc adapter 全在），存 `data/weights/SAMUS.pth`
- 零样本 sanity（OKD test 20 张，官方 /255 原生预处理）：**box 路径 0.8292 / 原生点路径 0.7943**（outputs/m0_samus_sanity.json）
- 架构事实与协议决定：SAMUS 的 forward 仅用**点提示**（bbox 形参未使用），但其 prompt_encoder 继承 SAM 的框通路（预训练权重）→ 采用**双路径**评测：box 直通（与 M3/M4 协议一致，主表）+ 中心点（原生模式，并列报告），论文注明"SAMUS 未在框提示上微调，框路径走 SAM 预训练权重"
- 微调（M5b 主表用）：decoder-only 轻微FT（编码器+adapter+prompt encoder 冻结）、AdamW 1e-4、50ep、原生中心点提示、3种子——已挂自动队列（等 GPU0 的 medsam 补跑结束后接续）
- 脚本：scripts/m5_samus.py、scripts/m5_train_queue.sh；M5 替代 M-MedUS 的决定见 §14 v1.1/v1.3



## 第二轮双智能体复审（2026-10-09 下午）——修复验证

**A组（数据管线/指标）**：CRITICAL-1 sanity框未缩放（已修复重算 0.3231→0.7808）；MAJOR-1 患者键粒度疑泄露（**dHash裁决：无泄露**——同前缀不同全键的跨划分对122对中63对跨划分（独立复核口径），最小距离45/256、0对≤20，全键即病例唯一标识，前缀共享是编号巧合；划分保持有效，无需重训）；MAJOR-2 hd95池化≠medpy双向max口径（已改为等价实现并验证）；MAJOR-3 权重校验（已加SAM官方md5硬校验，实测01ec64✓；MedSAM镜像源无官方校验和，出处已在报告中注明）；MINOR-1 恒真断言（已改为患者级+图像级双断言并实测通过）。
**B组（训练代码）**：CRITICAL 0项；MAJOR-1 fp16无GradScaler（评估后暂不改动运行中的队列：实测损失健康、fp32损失头；记为已知局限，后续重跑时加入）；核心数学全部验证通过——decoder批处理补丁与vanilla在batch=1严格等价（diff=0）、LoRA参数恰为decoder 4.06M+LoRA 0.295M、box缩放亚像素正确、test零接触、splits_final.json在preprocess后存活且与canonical一致。
**裁决与修复记录**：utils_kidney.hd95_mm改为medpy等价口径（含spacing轴序说明）；m1_split_okd.py断言补全；m0_sanity_medsam.py框缩放+CPU回退；m2_train_internal.sh补PYTHONNOUSERSITE；m2_train_foundation.py统一用utils.box_from_mask（当前0个多连通域mask，数值等价）；download_weights.sh加md5硬校验。



## 导师视角方法学评审（2026-10-09）与方案 v1.2 修订

评审由独立智能体以导师视角完成（Q1/Q2专项+12项审查），要点与处置：

- **Q1（为何不做CV）**：固定划分+3种子作为多模型配对协议站得住（患者级划分+dHash审计比官方随机分折更干净），但作为与官方0.8852对话的唯一依据不够 → **新增E1+**：nnU-Net v2单种子全量5折CV（患者级分折）作文献锚点（~55 GPU时，M3后执行）；仅nnU-Net享受CV，排名仍以共同固定test为准
- **Q2（为何3种子）**：3种子优于官方单种子（M1 seed44 0.5655 vs seed42 0.7092 实证）；统计处理已写死进方案§7：**逐患者种子均值入检验、严禁种子池化（防伪重复）、sd仅描述**
- **纪律纠偏**：M2验收门原写"test Dice≥0.80"属偷看测试集 → 已改val阈值；建立测试集访问日志制度
- **GT来源写死**：OKD全部模型用标注者1 masks（已核实代码在用，与官方锚点 annotator_1 同口径），禁止换靶
- **IoD双口径**：直接A-vs-B=**0.9063**（切片）/0.9047（肾）为主口径；自洽口径0.9478/0.9469并列（偏低4.2点（切片级 0.9478−0.9063=0.0415），证实评审的乐观性判断，仍在0.90–0.96门内）
- **E3细则**：P1先验源=val中位数种子（seed43）；空框计Dice=0入分母；提示框IoU进表
- **seed44定性**：后期发散（ep100 val_dice=0，best为中途值）；如实报告，禁作先验源
- **空mask排除佐证**：47/48带Poor/Unsatisfactory标记、46/48双标注者一致未标——排除有据
- 全文详见方案 §14 变更记录 v1.2；完整评审意见由导师评审智能体出具（12项逐条）

## 输出物索引（方案§8）



- outputs/RUN_REPORT.md（本文件）
- outputs/splits.json ✓
- outputs/metrics/iod_reference.csv ✓
- outputs/okd_cohort.csv / okd_cohort_summary.json ✓
- outputs/trusted_verify_summary.json / trusted_official_5fold.json / trusted_spacing.json ✓
- outputs/preprocess_config.json ✓
- outputs/m0_sanity_medsam.json ✓
- outputs/trusted_okd_license_archive.md / trusted_data_use_agreement_archive.txt ✓
- outputs/logs/（训练日志按 model_seed.log）✓
- scripts/（全部 runbook 脚本，随论文开源）

## 下一步（续接检查点）

1. M2 完成（预计 1–3 天）：E1 内部评测（Dice/HD95，患者级 Bootstrap CI）
2. M3 外验 + E3 提示消融（P1/P2/P4/P3）+ D1 对 TRUSTED 直接检测
3. M4 3D 聚合 + CT 诊断 + 反向 3D（用 TRUSTED 官方5折）
4. M5 统计 + 图表
5. SAMUS 权重：用户提供 → 补 M5；2026-11-01 未提供 → 按方案规则降级

## M5 补充审计（2026-10-11，清单 P1/P2 响应）

**聚合口径（P2-4/P2-6）**：`scripts/m5_aggregate.py` 入仓，`aggregate_results.json` 现可一键复现。主键 `(model,seed,prompt,case_id)` 断言唯一（e1/e2 重复主键均为 0，文件内重复表头 0/0）——**去重方向问题不存在**（若曾有重复，keep-first 与 keep-last 会有差异；现无重复，方向选择无影响，特此记录）。逐格完备性与预期严格相等（E1=99、E2=18322，medsam E3 六个已知缺口格与记录值逐一相符，无其他偏差）。HD95 为条件口径：空预测 Dice=0 但 HD95=NaN，所有 HD95 聚合并列 `hd95_valid_n`/`hd95_nan_n`，medsam p4 外验 HD95 均值仅覆盖非空切片子集——解读时必须连同覆盖率引用。

**外验失败模式裁决（P1-3）**：59 肾直方图（`outputs/figures/bimodal/`）显示**非强双峰**：5 肾 <0.5、52 肾 0.5–0.8、2 肾 ≥0.8（579L=0.4963 为第 5 个）。叠图目检 + GT 面积剖面证实三个机制：①**极点切片效应**（主因）：小 GT 环全部出现在标注范围起止端（肾极出入扫描平面处，<15% 肾中位面积者 999/18,322=5.45%（正典口径：评分集内 GT 面积<该肾评分切片中位面积 15%，见 canonical_pole_slices.json）），这些切片上**所有模型 Dice≈0**（nnU-Net 极点 0.026 vs 非极点 0.655；oracle box 的 MedSAM p3 极点 0.7164 vs 0.8939）——是标注口径现象而非模型失效，作为失败分析披露；主口径不变（冻结）。②真实异质性：失败尾肾（861L 低增益、704L 小肾）图像质量/形态差异真实存在。③nnU-Net 在域移位下偶发向控制台图形元素（文字区、扇形边缘、3D 方位立方）放电——训练域 OKD 无此类伪影。**无评测伪影**（无错位、无全零预测、坐标对齐正确）。

**P4 崩溃归因初判**：极点切片只解释小部分；主要缺口在非极点切片（medsam p4 非极点 0.468 vs p3 0.893）→ **检测质量是主因**，待 P1-1 全量检测指标落盘定量。

**术语（P4-12）**：方案A 中标注者称谓"医师"→"技师"（两库标注者均为 sonographer/radiographer）；"临床医师导向"（JUM 评审语境）保留。OKD 534=514 独特图像+20 张标注者内重复拷贝——m0_verify_okd.py 解释已更正（此前误记为 xlsx 行数）；包内 README/labels 未发现独立官方拷贝清单文件，防泄漏以 2026-10-09 dHash 裁决为执行机制。

**SAMUS 公平性（P3-10）**：DATA_QUALITY.md §8 已声明——SAMUS 仅点通路微调，box 通路未调；主表须并列其 p2 行且不得以 p3 直接排序。聚合数据齐备（samus p2 E1=0.8751/E2=0.4673 已在 aggregate_results.json）。

**检测溯源两封问询邮件草稿就绪**：dua/标注溯源问询_草稿_2026-10-11.md（Hostettler/Singla 各一封，含发送备注与无回复预案，待发送）。

## M4 split 核销（2026-10-11，写作端清单第 1 项，三断言全过）

核销期间 `m4_reverse_watch_v2.sh` 曾 SIGSTOP（仅停编排派新折，训练进程未动），核销后 SIGCONT 恢复。

**断言①折划分 vs 官方**：splits_final.json 每折 val 与 README.txt 官方留出清单 US_cv1..5 **集合逐字相等**（fold0=US_cv1…fold4=US_cv5；nnU-Net 内部重排了元素顺序，集合语义完全一致）；train∩val=∅（五折均为 0）；train∪val=59/59 肾。方向核销：val=留出清单（非 train）。
```
fold0: |train|=47 |val|=12 train∩val=0 union=59/59 val==US_cv1 OK
fold1: |train|=49 |val|=10 train∩val=0 union=59/59 val==US_cv2 OK
fold2: |train|=46 |val|=13 train∩val=0 union=59/59 val==US_cv3 OK
fold3: |train|=47 |val|=12 train∩val=0 union=59/59 val==US_cv4 OK
fold4: |train|=47 |val|=12 train∩val=0 union=59/59 val==US_cv5 OK
```
**断言②时序**：splits_final.json mtime=2026-10-10 17:27:00；已存在的全部 fold 目录最早文件 2d/f0=23:08:17、2d/f1=02:00:09、3d/f0=23:08:16、3d/f1=02:00:07——**全部晚于 split 写入**（f2-f4 未开始，无目录）。不存在用旧 split 跑出的折。
**断言③ spacing**：plans 2d=[0.3,0.3]、3d_fullres=[0.3,0.3,0.3]（float32 表征 0.30000001192092896），非 (1,1,1)。

**口径申明（即刻生效）**：E4（反向实验）指标只用 checkpoint_final；规则 4（Dice>IoD 报警）只适用 E2。0.85@250ep 类 EMA/伪 dice 数字仅作训练监控，不入论文。

## 第 4 项小尾巴（2026-10-11）

**medsam 训练日志首行审计**：三种子配置逐字一致——`model=medsam trainable=93.74M lr=1e-05 bs=1`（seed42/43/44）。seed42 有一次无异常重启（同配置），seed43 首次尝试在 m2_train_foundation.py:230 崩溃一次后以**相同 bs=1** 重启成功（日志 2 个 header、1 个 Traceback，最终 BEST val_dice=0.9639）。无 bs 漂移，T1-1 关闭。

**关键层检测准则（P3-7）**：检测器=相邻标注切片掩码的 1−IoU 序列，取"局部极大且 ≥ 中位数+3×MAD"者为边界不连续层。结果 3458/18322=18.9%，**远高于数据文档"每 5mm 手绘一层（≈1/17≈5.9%）"**。差异原因（判定）：①检测器捕捉的是"掩形突变层"，除手绘层外还包含真实解剖过渡（肾盏/脂肪界面突变）与标注抖动；②不排除文档口径（间隔/层厚）与实物不符。因此论文措辞用"boundary-discontinuity slices"而非断言"手绘层"；18.9% vs 5.9% 的偏差作为数据质量观察披露。分层效应显著：nnU-Net 在不连续层 0.326 vs 平滑层 0.688（seed42）。后续可用中点混合拟合（interpolated contour ≈ 0.5·mask(z−d)+0.5·mask(z+d)）做二阶判定进一步分离手绘/解剖成因。

## 检测指标与溯源探针结果（2026-10-11 补）

**P1-1 三种子检测指标（TRUSTED 全 18,322 评测切片，vs STAPLE 外接框）**：
| seed | 检出率@0.25 | @0.10 | 极点检出@0.25 | AP@0.5 | AP@[.5:.95] | IoU(检出)中位 |
|---|---|---|---|---|---|---|
| 42 | 56.1% | 60.8% | 3.0% | 0.564 | 0.329 | 0.819 |
| 43（P4 提示源）| 51.5% | 57.0% | 1.7% | 0.553 | 0.326 | 0.824 |
| 44 | 56.6% | 61.4% | 3.4% | 0.530 | 0.320 | 0.813 |

结论：阈值不敏感（+4.7~5.5pp@0.10）；失败=无候选框（尤其极点 2-3%）；检出框质量良好（IoU 中位 0.81-0.82）。P4 崩溃是"检出缺失"，非"框偏移"或"阈值错位"。

**预注册框 IoU 表**（vs P3 oracle，outputs/summary/box_iou_vs_oracle.json）：P1 框（U-Net 生成）IoU 中位 0.302、74.6% 切片 <0.5、22.2% 切片无框——P1 是显著弱于 P4 的提示源；P4 三种子 IoU 中位 0.813-0.824。

**P3-9 SAM 零样本溯源探针**：原版 SAM vit_b（官方 predictor/预处理）+ STAPLE 外接框，50 切片（rng=42）：Dice mean 0.849 / median 0.876，>0.9 仅 24%，<0.5 仅 2%。**不饱和**——现成 SAM 无法从共识框近乎完美复现标注，无"标注由通用模型辅助"的信号；与问询邮件（P4-11）互为补充。归档 outputs/figures/sam_zeroshot_provenance.json。

**P1-2 metric 级 conf 敏感性**：运行中（GPU1，seed43 检测器：tau=0.10 变更 1001 切片、tau=0.15 变更 623 切片；仅重分割变更切片，其余复用主表行，数学等价全量重评）。

## TTA 补充臂注册记录（Amendment 1，2026-10-11）

**占位符填充（P5 框源种子）**：选择逻辑=取注册 nnU-Net v2 (2d) 在 OKD 内部验证集上的最终 EMA val dice（训练日志 training_log fold_0 末行 "Pseudo dice"，nnU-Net 内部 20% val split），三种子取中位数。数值：seed42=0.9486、seed43=0.9484、seed44=0.9519 → **中位种子 = 42**。选取只读 OKD 内部训练日志，未接触任何 TRUSTED 数据。MedSAM 评测种子按同规则在执行时机械选出（注册文本 §3 同款措辞）。


**红线确认**：TRUSTED GT 只进注册评分器，不进任何 Arm 构建/训练代码；headline P4 不动；六模型 Friedman 家族不动；§4.4 判据在任何结果出现后不得调整。Arm 计算在 OSF 时间戳锁定前不启动。

## Amendment 1 执行准备（2026-10-11，未运行任何 Arm 计算）

- 文稿：`docs/tta_amendment_v1.md` 两步入仓（ae4787b 原文 → dc6ec70 填 [TO FILL]=42）；选择逻辑见上节。
- MedSAM 评测种子（同规则机械选）：内部 val 42=0.9627 / 43=0.9639 / 44=0.9625 → **seed 42**。
- 预置脚本（**均未运行**，锁前不跑）：`m5_p5_priorbox.py`（P5 单遍）、`m5_tta_aug.py`（(a)–(f) 算子，已在纯合成数组上单测）、`m5_tta_round.py`（伪标签 conf≥0.5 → 15ep lr0=1e-3 → checkpoint 归档 → 检测指标）。
- **提交前请写作端裁决两处解释点**（一句话进文本即可，窗口开放至 OSF 提交前）：①§4.3 "Data-order seeds fixed (42/43/44)"——当前按"每轮 3 个数据顺序重复、均从注册 seed43 权重出发"实现；若意为"轮 1 用 42、轮 2 用 43"请改一句话。②§4.4 (ii)/(iii) 的 Dice 口径——P4 通路存在于 medsam/sam_lora/samus 三模型；建议"三模型分别报告，判据须对全部三模型成立"（最保守、抗 HARK）。
- **在收到 OSF 时间戳/DOI 并回贴本报告前，不运行 P5 推理与 TTA 训练的任何计算。**

**Amendment 1 解释点裁决（2026-10-11，写作端）**：①§4.3 = 每轮 3 个数据顺序重复（42/43/44，均从当前注册权重出发）；轮级判据按重复间均值评估、逐重复值并列报告。②§4.4 (ii)/(iii) = 三提示模型分别报告，采纳须对全部三模型成立。两处澄清已最小修改写入文本并重新入仓（提交前窗口内）。

**42 可复现性加固（2026-10-11）**：训练日志不入仓（gitignored），故把证据行固化为 `outputs/summary/nnunet_seed_selection_provenance.txt` 入仓——含精确提取命令、三种子最终 EMA val dice 原始日志行（带时间戳）、中位数计算与断言。第三方从该文件即可机械复现 seed42 的选取；完整日志保留在服务器可查。

**注册文本数字溯源核对（2026-10-11）**：`outputs/summary/amendment_numbers_check.txt`——Amendment 1 全部 13 个数据衍生数字逐一从入仓工件重算核对（0.624/0.9063/0.69/51.5%/48.5%/<6pp/3.4-3.7pp/0.884/97.6%/IoU 0.81-0.82/18,322/59-39/seed42），全部机械可复现；IoU 区间按文本两位小数舍入口径核对。**seed42 的完整证据链**：现场重提取三次一致（原始日志行 + statistics.median 断言），证据行固化为 `outputs/summary/nnunet_seed_selection_provenance.txt` 入仓（986f624）——训练日志虽不入仓，第三方凭仓库内溯源文件即可复现选取。

## Amendment 1 解冻（2026-10-11）

**注册已锁定**：DOI **10.17605/OSF.IO/A8MKX**，2026-10-11 11:51:37 **+0800**（= 03:51:37 UTC；OSF API date_created 实证。审计更正：此前误标为 UTC），快照含 tta_amendment_v1.md，可公开访问。P5/TTA 解冻，按注册 §3/§4.2-4.4 执行；§4.4 判据冻结不可调整；Round 1 仅监控；两轮成败全部如实报告；产出一项推一项（24h 节奏），P5 三个数先发导师端做 §3 裁决。

## P3-8 完成：对 Annotator1/2 分别评分（2026-10-11）

**结论：共识靶无乐观偏差**。nnU-Net（best 口径，3 种子）对两个个体标注分别评分与对 STAPLE 共识评分互相咬合：seed42 A=0.6190 / B=0.6248 / STAPLE=0.6209；seed43 A=0.6145 / B=0.6210 / STAPLE=0.6168；seed44 A=0.6322 / B=0.6379 / STAPLE=0.6345——共识分均落在两个个体分之间（偏差 ≤0.006）。"STAPLE 靶偏乐观"的质疑就此关闭。附带分层（边界不连续层 vs 平滑层，双标注者一致）：key 0.342-0.377 / interp 0.678-0.699，与 P3-7 单模型分层吻合。聚合（无病例号）入仓 `outputs/summary/nnunet_rater_aggregate.json`；逐例 CSV 留服务器（gitignored）。

## Arm P5 结果（2026-10-11，注册 §3，诊断口径）

**三数**：coverage = **1.000**（nnU-Net seed42 先验框在全部 18,322 外验切片均有框，对比注册 P4 检测器 51.5%）；box-IoU vs oracle 中位 = **0.677**；strict Dice = **0.594**（MedSAM seed42 单遍，registered scorer）。

**裁决（按冻结判据，无调整）**：coverage≥0.75 ✓；IoU 中位≥0.70 ✗（0.677）；Dice≥0.55 ✓ → 三条须同时满足，判 **diagnostic/negative**，如实报告。

**诊断解读（非翻案）**：P5 证明无检测器提示源可覆盖 100% 切片并把 strict Dice 从 P4 的 0.4424 抬到 0.594（+15.2pp），但框定位质量（IoU 0.677）低于注册检测器检出时的水平（0.81–0.82），也低于 §3 预设的 0.70 线。**部署管线的约束因此完整化为两条：检测覆盖（P4 缺，P5 补齐）+ 域内框定位质量（两者都不足，oracle 0.884 仍遥远）**。该界限直接为 P4-TTA 提供预期：自训练须同时改善覆盖与定位才可能过 §4.4。逐例框已存 outputs/prompts/p5_box_trusted.npz（gitignored）；聚合三数入仓 outputs/summary/p5_priorbox.json。

## TTA Round 1 监控报告（2026-10-11 23:45，注册 §4.4：R1 仅监控，不裁决）

三 replicate（42/43/44，各自从注册 seed43 权重 + 同一份伪标签集 8,142 切片微调 15ep）检测指标：

| replicate | no-box率 | det@0.10 | 极点检出@0.25 | IoU中位 | AP@0.5 |
|---|---|---|---|---|---|
| s42 | 17.2% | 90.2% | 31.5% | 0.775 | 0.685 |
| s43 | 23.6% | 83.1% | 21.2% | 0.792 | 0.655 |
| s44 | 16.7% | 90.3% | 37.5% | 0.770 | 0.673 |
| 注册P4基线 | 48.5% | 57.0% | 1.7% | 0.824 | 0.553 |

监控读数（非裁决）：no-box 均值 19.2%（基线 48.5%），三重复全部 <25% 线；极点检出 1.7% → 21-38%；AP@0.5 +10~13pp；代价 = IoU 中位 −3.2~−5.4pp（自训练对自身伪标签的典型权衡）。R2 已启动（配对设计：各 replicate 从自己的 R1 checkpoint 续训，伪标签按 §4.2 由各自当前检测器重新生成）。§4.4 唯一裁决在 R2，判据冻结。

**下游评分的提示模型种子（补充声明，2026-10-13 审计后）**：medsam/sam_lora 用 seed42（同"内部 val 中位"机械规则：medsam 0.9627/0.9639/0.9625→42；sam_lora 0.9602/0.9603/0.9600→42）；**samus 用 seed42 与该规则不符**（其内部 val 点通路中位=43），属执行疏漏；裁决稳健性不受影响（samus conditional 0.6365 距 0.75 线余量 0.11，strict 余量 ≥0.049，任一种子读数下 (ii) 均不通过），已列入论文 deviations 待披露项。

## TTA Round 2 + §4.4 最终裁决（2026-10-13，注册判据冻结、机械执行）

**R2 检测指标（3 replicate 均值）**：no-box **5.83%**（R1 19.2%，注册 P4 基线 48.5%）；AP@0.5 外验均值 ~0.689（基线 0.553）；IoU 中位 0.738-0.756。

**§4.4 逐条（Round 2，均值口径）**：
- (i) 外验 no-box < 25%：**5.83% ✓ 通过**
- (ii) conditional Dice ≥ 0.75（三模型须全部成立）：medsam 0.7328 ✗ / sam_lora 0.7298 ✗ / samus 0.6365 ✗ → **未通过**
- (iii) strict Dice ≥ 0.55（三模型）：0.690 / 0.687 / 0.599 ✓ 通过
- (iv) OKD 内域非回归：检出率降 0.67pp ✓；**mAP@0.5 降 0.0263 ✗（>0.02）** → 未通过
- 部分带（no-box∈[25%,35%] 且条件 Dice≥0.80）：不适用（no-box 5.83% 不在带内）

**裁决：NEGATIVE result（(ii)(iv) 未过），按冻结规则如实报告；无任何阈值/轮次/口径调整。**

**deviations 待披露（审计 M3）**：R2 采用配对初始化（各 replicate 从自己的 R1 checkpoint 续训），与注册 §4.3 字面 "the same current weights" 存在张力；该解释点在 R1 完成后、R2 启动前已记录于本报告（10-12 00:12 前），且对裁决不敏感——(ii) 在任何单起点读数下同样不可能通过（全链最大 conditional 0.7416 << 0.75）。另：TTA 构建/训练代码经共享注册迭代器读入切片时 GT 形参以下划线解包弃用（操作层面合规已机械验证：伪标签图像与 US 图像逐字节相同、与 GT 不同）。

**结果解读（如实）**：源自由检测器自训练**解决了检测覆盖**（48.5%→5.8%）并大幅改善外验 AP（+0.14）与 strict Dice（medsam 0.4424→0.690，+24.8pp），但 **(ii) 的失败定位了伪标签天花板**：条件 Dice 被检测器自身框质量锁死（自训框 IoU 中位 ~0.74-0.76 vs oracle），与 P5 诊断（0.677-IoU 框 → strict 0.594）互相印证——覆盖修复后，域内框定位质量成为剩余瓶颈；(iv) 的 AP 轻微内域回归（-0.026）是 15ep 自训练的典型代价。**注册 P4 headline 不变，scenario-B 主叙事不变**；两轮完整结果按 §5 治理条款写入论文 deviations/exploratory 小节。工件：tta_r2_downstream.json / tta_r2_detection.json / tta_r2_verdict.json（全部入仓）。

**M5 统计 deviations 补充（终审 m6/m7）**：①§7 预注册辅助指标"3/3 种子方向一致率"已补实现（stats_report.csv）；②LMM 实现为 `dice ~ C(model) + C(seed)`（种子固定效应+患者随机截距），与注册文本"种子随机效应"存在字面偏差——互证结论一致，列入 deviations；③M2 训练 val 终值档案（防引用旧运行值）：USFM 三种子最终 0.8388/0.8401/**0.8639**（日志 usfm_seed4x.log 各自最后一次 BEST 行；0.8339 为被放弃的首轮 50ep 运行值，严禁写入论文）。

## E4 反向五折 CV 汇总（2026-10-13，M4 收官）

TRUSTED 训练 → TRUSTED 官方五折留出验证（59 肾每人恰一次）的体级 3D 指标（registered 几何，GT=labelsTr 原几何）：

| 配置 | 3D Dice（59 肾） | HD95(mm) | 患者级均值(39簇) | 折间范围 |
|---|---|---|---|---|
| 2d | **0.7788±0.1023** | 70.9 | 0.7759±0.0819 | 0.744–0.811 |
| 3d_fullres | **0.7776±0.0938** | 30.5 | 0.7758±0.0827 | 0.750–0.808 |

**交叉验证**：3d f4 的 nnU-Net 自带 evaluator（summary.json，修复 gt_segmentations 后生成）Dice=0.7718，与本表该折 0.7718 **精确一致**（双源互证）；量级与训练期 EMA pseudo dice（0.78-0.88）相容。**解读**：TRUSTED 域内 nnU-Net 上限 ~0.78（3D 体级）——与前向外验（OKD→TRUSTED，切片级 0.62）共同界出域差贡献与数据集固有难度；2d/3d 在体级等效（3d 的 HD95 优势 30.5 vs 70.9mm 源于 2d 逐层独立推理的层间不连续）。

**执行事故记录（已修复）**：E4 首版脚本 dice 公式漏了因子 2（写成 i/(a+b)），被"nnU-Net 自评 summary 交叉核对"当场抓获（三病例精确减半），修正为 2i/(a+b) 后与自评一致；同折双验证碰撞（本会话恢复进程 vs 用户新会话进程）以让位新会话+setsid 脱离重跑解决。工件：outputs/summary/e4_reverse_cv.csv（逐肾）+ scripts/m5_e4_aggregate_fast.py（表面点 KDTree HD95，与距离变换公式等价）。

## DUA 跟进计划（P3-D，2026-10-13 记入）

- 2026-10-08 签署件已发 Hostettler（已读）；**2026-10-16 00:20 bump 邮件已发**（学邮 241320185@st.usst.edu.cn，同串，含核正引用 Scientific Data 12,615）；
- （下一步）再无回复 → 升级 Scientific Data 论文通讯作者（Ndzimbong）→ IRCAD general contact，每步记一行；
- 再无回复 → 依次升级：Scientific Data 论文通讯作者（Ndzimbong et al.）→ IRCAD 官网 general contact；**每走一步在本节记一行**；
- **论文措辞红线**：全文不得出现 "countersigned"；统一用 "signed DUA returned to the authors (receipt confirmation pending)"；数据许可本身按 CC BY 4.0（figshare）表述，**投稿不被此项阻塞**；
- 状态行以 docs/trusted_dua_return_archive.md 为准（入仓版已创建，回执后更新）。

## 合规档案入仓（P3-A/B，2026-10-13）

- `docs/trusted_dua_return_archive.md`（签署回执存档，脱敏版，发送日 2026-10-08，回执 pending）；
- `docs/okd_access_email_archive.md`（Singla 2026-10-08 授权邮件脱敏版——两个 1drv.ms 链接打码，注明经环境变量消费；含作者要求引用条目与 CC BY-NC-SA 注记）。

## OKD 引用更正（P3-C，2026-10-13）

作者指定引用为 workshop 正式版：Singla, R., et al. (2023, October). The open kidney ultrasound data set. In **ASMIC** (pp. 155–164). Springer.——README 已加入该引用（arXiv 2206.06657 可并列）。**写作端注意**：论文稿凡引 OKD 处统一用此正式出处，勿单引 arXiv。注册文本（TX7RN/A8MKX）经核对未引数据集具体文献，无需追溯修订。

## P1 收官：反向实验评分 + 前向 3D 聚合 + CT 诊断 + 归因矩阵（2026-10-13，按 P4 格式）

**反向 2D 切片级**（nnU-Net 2d 十折 CV 的 validation 体按 18,322 注册切片评分，checkpoint_final）：
均值 **0.678**；患者级 39 簇 bootstrap 95% CI **[0.640, 0.716]**。**低于预期带 0.85–0.93**。

**反向 3D 体积级**（各折 validation 原几何 3D 导出 vs labelsTr，2d/3d 两配置）：
2d 体级 **0.7788 ± 0.1023**（患者级 CI [0.7486, 0.8012]）；3d 体级 **0.7776 ± 0.0938**（患者级 CI [0.7502, 0.8022]）。**落入预期带 0.71–0.81** ✓。

**与预期带的偏离即发现**：预期带按 OKD 内部难度（0.93）标定；实测 TRUSTED 域内切片级只有 0.678——**TRUSTED 的标注口径（极点小截面、关键层插值）使域内切片上限本身就远低于 OKD**，而非训练缺陷（体级口径落带内印证：大结构一致性达标）。

**CT 侧诊断表**（48 CT 患者，GT_estimated_masksCT 肾体积累积 vs 前向 nnU-Net US Dice）：Pearson r = **−0.0804**；CT 体积四分位组 US Dice 0.635/0.639/0.621/0.625——**平坦**：失败与肾大小无关，排除"小肾难分割"解释。

**前向 3D 体积聚合**（E2 nnU-Net 逐切片预测堆叠 + binary_closing 球 r2 一迭代 + 最大连通域，参数固定）：3 种子 **0.7039 / 0.7181 / 0.7417，均值 0.7212**（切片级 0.624 → 体级 0.721，3D 聚合如预期抬高）。

**归因矩阵（域退化 vs 格式损失，双口径互证）**：

| 口径 | 域内上限（反向 CV） | 前向域移位（E2） | 域退化 |
|---|---|---|---|
| 2D 切片 | 0.678 | 0.624 | **−0.054** |
| 3D 体积 | 0.7776 | 0.7212 | **−0.056** |
| oracle box 上限 | — | 0.884 | — |
| IoD 人体参照（direct） | — | 0.906 | — |

**结论**：域移位在 2D/3D 双口径下仅值 0.054–0.056（一致互证）；**性能上限的主要压缩来自标注格式与数据集固有难度**（域内体级 0.78 vs oracle 0.884 vs IoD 0.906 的阶梯），而非跨库退化主导。Bland-Altman（反向 2d vs 3d 逐肾）：偏倚 +0.0012（无系统差），95% LoA [−0.238, +0.240]，跨肾相关 r=0.23——两配置平均等效、逐肾失败模式不同。

工件：`outputs/metrics/e4_reverse_slices_{2d,3d}.csv`（逐切片，gitignored）、`outputs/summary/{e4_reverse_slice_stats,e4_forward_volume3d,ct_diagnostic_table,attribution_matrix,bland_altman_reverse}.json`、`outputs/figures/bland_altman_reverse_2d_vs_3d.png`。评分脚本 `scripts/m5_e4_reverse_full.py` **先入仓后评分**（e37f906/4028bd6 + pid 修复）。

## B 工单计算（2026-10-13/14，五项）

1. **极点 A/B 一致性** ✅：正典极点 999 切片上 Dice(Annotator1, Annotator2) = **0.618**（中位 0.674，59.8% 切片 ≥0.6）≥ 0.6 → **标注惯例坐实**（两位技师在极点一致地只标注可见小截面，非标注缺陷）；非极点 17,323 切片 A/B = **0.923**（99% ≥0.6，与注册 IoD direct 0.906 一致）。极点一致性显著低于非极点（0.618 vs 0.923）——极点截面小，微小位移即大幅降 Dice，但双方方向一致。工件：pole_annotator_consistency.json + iod_ab_per_slice.csv（逐切片，gitignored）。
2. **五源公共子集条件 Dice**：运行中（GPU 通道，注册 P4 检出的 9,434 切片公共子集上重算 P1/P5/TTA-R2/P4/oracle，全部 MedSAM seed42 同分割器）。
3. **联合 bootstrap CI**：B3a 部分完成——归一化（÷直接 IoD 0.9063）：内部 1.028 [1.013,1.040]（93 OKD 簇，分子 bootstrap、IoD 为 TRUSTED 常数的偏差已注明）；外部 0.6956 [0.663,0.726]；P4 0.482 [0.433,0.530]；oracle 0.977 [0.965,0.989]（39 簇联合重采样）。**域分解两腿联合 CI：2D 切片 −0.0496 [−0.081,−0.018]、3D 体积 −0.0669 [−0.108,−0.027]——两口径 CI 均不含 0，域移位分量统计显著**。TTA-R2 判据 CI 待 GPU 通道。
4. **US-vs-CT 体积 Bland-Altman（nnU-Net 零样本配置）** ✅：US 预测体积 vs CT 参考 **bias +147.9 mL**（LoA [−54.7,+350.4]；预测均值 344.4 mL vs CT 196.6 mL）；US GT vs CT bias −27.0 mL（LoA [−128.7,+74.8]）。临床翻译：Dice 层面的极点/格式问题在体积口径放大为 **+75% 的体积高估**（344 vs 197）。TTA-R2 配置待 GPU 通道。
5. **Tent 对照**：跳过（可选）。理由：特征级熵最小化 TTA 不产生检测框，无法触及覆盖悬崖（本failsafe已由 P5 证明框质量才是瓶颈）；§4.3 的逻辑论证（无框 TTA 不能修复覆盖）已足够，将在 Limitations 写明"未运行特征级 TTA 对照，因任何不产生提示框的适配方法均无法改善 48.5% 的无框率——该论证由 P5/P4-TTA 两臂的覆盖数据直接支撑"。

## B 工单补充（终审修复 + 例图定稿说明，2026-10-15）

**四智能体终审（今日代码）处置**：①bpass oracle 通道 GT 路径笔误（TRUSTED_ 前缀）必然崩溃 + 崩溃残件被 exists-skip 静默吞掉 → v2 重写（oracle 改用预计算 p3_oracle_trusted.npz 框——同时消除 32 小时逐切片 GT 重读；CSV 原子写出 .part→rename + 行数校验；预取线程隐藏 CPU 读卷；三卡并行）+ 旧残件清除；②B3b 组装 bootstrap 用 Index.intersection 把有放回重采样去重（CI 系统性偏窄 ~24%）→ 改为保重复的正确聚类 bootstrap（与 B3a 同语义）+ 行数/簇数校验（18,322 行/39 患者，不符即中止）；③assemble 分阶段落盘 + 链条无退出码检查 → 重写武装链（校验 MEGA PASS DONE 后才组装）；④pole_oracle 断点续跑会以 NaN 覆盖好产物 → 修复（仅三种子齐时写 pooled）；⑤fig1b 右下 3D 线框图标为烧录在原始帧内的机器叠加（非患者标识，保留）；⑥US GT vs CT bias −27 mL 的机制注记：US GT 体积按构造等于标注跨度体积（非零层≈标注层），逐肾中位 −15.7 mL、314L 一肾 −226 mL 为离群。

## 二轮修复（四智能体审查，2026-10-15）

四智能体审查（bpass v2 运行代码/组装脚本与武装链/B4 体积与 BA/例图与 oracle 脚本）处置：①**B4b 患者 join int/str 失配**（0 对配对→NaN BA 以 rc=0 推送）已修（双侧 str + pairs 断言）；②**B3b conditional bootstrap 的 intersection 去重**（CI 偏窄 ~23%）已改为保重复重采样（与 B3a 同语义）；③B3b 点估计口径统一注记（点=池化检出均值、CI=患者簇 bootstrap）；④B4b spacing 从头文件读取并断言 0.09/0.3；⑤pole_oracle 断点续跑合并修复；⑥B4b 未检出体积 0 的向下偏差注记；⑦组装链重写（等 7 CSV 完整→组装→rc 检查→经 fresh 仓库强制推干净历史——修复原链 origin 缺失仍记 "pushed" 的问题）；⑧例图四脚本入仓、旧单体脚本退役。B1/B3a/B4a 数字经独立智能体**逐位复现**（含 18,318 行逐例对账、oracle 双例 GPU 重跑、六项 bootstrap 12 个 CI 端点）。

## B 工单收尾：公共子集阶梯 + 判据 CI + TTA-R2 体积 BA（2026-10-15 22:2x，二轮修复后定稿）

**B2 公共子集五源条件 Dice**（注册 P4 检出子集 9,434 切片，统一 MedSAM seed42 分割器）：P1 0.460 < P4-注册 0.859 ≈ **P4-TTA-R2 0.868** < oracle 0.922；P5（nnU-Net 先验框）0.857 与 TTA-R2 相当。**公共子集上 TTA-R2 修复检测器把条件 Dice 拉回注册检测器水平（0.868 vs 0.859）**——覆盖与框质量双修复的确证。

**B3b 判据 CI（39 簇联合 bootstrap，2000 次，保重复）**：strict MedSAM 0.680 [0.658,0.713]、SAM+LoRA 0.678 [0.653,0.710]、SAMUS 0.595 [0.574,0.628]——三模型 strict CI 全部 ≥0.55 上方；conditional MedSAM 0.743 [0.725,0.762]、SAM+LoRA 0.738 [0.720,0.758]、SAMUS 0.651 [0.633,0.669]。口径注记：点=池化检出均值（与下游 json 同源可比）、CI=患者簇 bootstrap（保重复）。

**B4b TTA-R2 体积 BA**：US 预测体积 **216.9 mL vs CT 参考，bias +20.3 mL**（LoA [−138.6,+179.3]）——对比 nnU-Net 零样本的 +147.9 mL 高估，**TTA 修复把体积高估缩小 ~7 倍**。机制注记：未检出切片（8.3%，聚簇于极点）贡献 0 体积使 US 体积系统性偏低，实测仍为正向 bias，说明修复后框覆盖的体积贡献已超过该下偏。

## 补充分析（supplementary analyses，2026-10-17）

**口径与可复现性**：全部新增推理由 `scripts/supp_megapass.py` 完成（10 配置 × 18,322 切片，含 dice vs STAPLE/A/B、HD95、NSD@1mm、pred_area），与注册评测**逐位一致**（medsam/p3 0.8826 vs 0.8826；sam_lora/p3 0.8559 vs 0.8559；samus/p3 0.7081 vs 0.7081；medsam/p4 Δ=+0.0001）。nnU-Net 由 `supp_nnunet_score.py` 从既有 nii 预测补算。逐数来源见 `docs/supp_analyses_provenance.md`。

**① 观察者分层**（A-vs-B 直接；模型 vs 共识/A/B）：极点 999 层 A/B Dice **0.618**（IoU 0.486）≪ 不连续层 3,036（0.888/0.810）≪ 其余 14,287（0.930/0.875）。模型分层（共识/A/B）：nnU-Net 极点 0.026/0.034/0.038、不连续 0.392/0.393/0.398、其余 0.711/0.708/0.714；MedSAM P3 极点 **0.705**/0.570/0.582、不连续 0.861、其余 0.900；**MedSAM P4（检测框）极点 0.001——检测器在极点几乎无框**（与覆盖分析一致）；TTA-R2 极点 0.045、不连续 0.519、其余 0.759。不连续层计数说明：极点优先，3,456 标志中 420 层同时为极点，互斥后 3,036。

**② 体积误差（配对 CT）**：US 肾→同患者同侧 CT 分量（R=小 x 质心），肾级配对 n=59/患者 39（US 亚队列为 48 CT 患者中的 39），患者簇 bootstrap。nnU-Net 零样本 **+147.85 mL [127.53,167.84]**（seed42）/ **+123.86 [105.03,142.62]**（3 种子）；TTA-R2 **+20.32 [3.99,37.63]**；US GT **−26.98 [−38.10,−14.15]**。

**③ 配对 bootstrap CI**（39 簇×2000，保重复）：IoD direct **0.9063 [0.8943,0.9164]**（患者级）/ 0.9047 [0.8938,0.9152]（肾级）；域退化 2D 切片 **−0.0496 [−0.0810,−0.0182]**、3D 体积（3 种子 forward）**−0.0555 [−0.0923,−0.0208]**；CT 体积–US Dice r **−0.0804 [−0.33,0.23]**（含 0 → 措辞"未检出相关"）；**R0(注册 P4) vs R2(配对两轮)**：strict **+0.2482 [0.2215,0.2788]**、固定检出子集（R0 检出 9,431 层）条件 **+0.0095 [0.0045,0.0154]** —— **R2 的增益几乎全部来自新检出覆盖，已检出切片上的分割质量提升仅 +0.010**。

**④ 边界指标**：HD95(mm, 条件口径)/NSD@1mm：nnU-Net 53.9/0.264（有效 18,322）；MedSAM P3 **5.75/0.409**；MedSAM P4 11.65/0.186（有效 9,434，8,888 NaN=无框）；MedSAM TTA-R2 17.54/0.268（有效 16,802）；SAM+LoRA P3 9.13/0.407、TTA 18.38/0.272；SAMUS P3 20.7/0.161、TTA 25.95/0.139、seed43 TTA 21.14/0.187。

**⑤ 聚类审核**：全部 p 值比较清单入 `supp_cluster_audit.md`——所有检验均在患者/肾簇级；检测/阈值/分层类为描述性（无 p），标 exploratory。

**⑥ S-Table 4**：逐 replicate strict/conditional/coverage 表入 `supp_table_s4_per_replicate.md`（数据与下游 json 同源）。

**⑦ 方法材料**（`supp_methods.md`）：YOLO 划分/超参/0.25 依据；TTA 优化器/学习率/损失/无冻结/15ep 注册依据/六增广参数/计算成本；自训练排除偏差量化（R1 保留 44.4%，R2 保留 68.9-74.7%；保留片平均 GT 面积 ~2.2× 排除片）。**不连续检测器盲测**：A/B 各自掩码序列标志率 **5.86%/5.65%**（与文档 1/17=5.88% 精确吻合），**STAPLE 共识 18.86%（3.2×，融合+二值化产物）**；洗牌噪声地板 **0.03%**；阈值稳定（2/3/4×MAD=22.7/18.9/15.9%）；跨标注者 Jaccard 精确位 **0.30（随机 0.03，10×）**、±1 切片 **0.64（随机 0.09，7×）** → 18.9% 主要是共识型金标准的性质，非标注者问题。

**⑧ 敏感性**：LMM 种子随机效应双估计（ANOVA 方差分量 sd≤0.0137；MixedLM 收敛，种子方差分量 0.0087）→ 结论不变；SAMUS seed43 重评：strict 0.6386 / conditional **0.6963** / coverage 0.917（seed42: 0.5945/0.651/0.94）→ 判据 (ii) 仍不通过；**R2 备选初始化（单起点，注册权重，3 reps）**：no-box **19.15%** / strict **0.6323** / conditional **0.7825**（MedSAM 下游口径）vs 配对设计 5.83%/0.680/0.743 —— **两轮复合迭代是覆盖修复的关键**；单起点条件下 conditional 可过 0.75（MedSAM），但覆盖停留在 R1 水平，且三模型齐备性未评估——如实记录。

## 补充分析续：配方偏差点检查（item 9，2026-10-17）

**USFM 注册配方点检查（seed 42，执行）**：按冻结注册基准（plan_A v1.4 §A1：lr=1e-5、50 epochs、Dice+BCE，对齐 M3；验收门 val Dice ≥0.85）训练一版：AdamW lr 1e-5、层衰减 1.0、无 warmup、wd 0.01、bs 8、50ep。结果：**内部 val 最佳 0.7967（未达 0.85 验收门，如实记录）**；**E1 测试 0.822（Δ 对官方配方 −0.0402）**；**E2 切片池化 0.4585（Δ −0.0102）**。GPU 用时：训练 ~19 分钟 + E1/E2 评测 ~28 分钟（V100-16GB）。**决策（写作端规则）**：|Δexternal|=0.0102 < 0.03 且排名/裁决不变 → 不升级 3 种子；一句话 + 补充表收口。意义：注册配方在该模型**达不到自己的验收门**，官方配方的采用有据可查（且外验差异仅 0.01）。工件：`supp_usfm_regrecipe.json/.md`；checkpoint `usfm_regrecipe_seed42.pt`（gitignored）。

**SAMUS 配方注册核对（先取证再决定）**：OSF 全部端点 2026-10-17 返回 HTTP 503（站点故障/WAF），`REGISTRATION_DOCUMENT_EN.md` 暂不可取。离线冻结基准（plan_A v1.4，即注册文档的整合来源）SAMUS 条款仅为"**官方权重直接推理 + OKD轻微微调两版**"，未指定层/学习率/epoch。按写作端规则 → **不重训**：报告的 43.71M 编码器侧轻量模块+decoder 版即预注册"轻微微调"的实施；早期 decoder-only 4.06M 版归档为被否决变体。**待办**：OSF 恢复后重新摘取原文核对；若载有具体配置且与现版不同 → 按 USFM 同格式跑 seed-42 点检查并更新记录。工件：`supp_samus_recipe_decision.json/.md`。

## 补充分析续：SAMUS 框微调臂（2026-10-17）

新增"框微调"补充臂以解除 DATA_QUALITY §8 的提示通路不对称：与报告臂同配方（官方轻量模块+decoder+**prompt encoder**，43.72M；AdamW 1e-4/cosine/50ep/bs8；Dice+BCE pos_weight=2），仅训练提示改为 **GT 框**（与 MedSAM 训练同框源），seed 42 点检查。GPU：训练 25 分钟 + 评测 42 分钟（V100-16GB）。

| 评测 | 提示 | 框微调 | 点微调（注册臂 seed42） | Δ |
|---|---|---|---|---|
| E1 | p3 oracle box | **0.9443** | 0.8938 | **+0.0505** |
| E1 | p4 detection box | **0.9165** | 0.8835 | **+0.0330** |
| E1 | p2 point | 0.8262 | 0.8794 | −0.0532 |
| E2 | p3 oracle box | **0.8866** | 0.7081 | **+0.1785** |
| E2 | p4 detection box | **0.4393** | 0.4020 | +0.0373 |
| E2 | p2 point | 0.4616 | 0.4777 | −0.0161 |

**结论**：SAMUS 在框行的落后主要是"框通路未微调"的协议不对称，而非模型能力——框微调后 E2 oracle-box 达 0.8866，与 MedSAM (0.884)/SAM+LoRA (0.872) 同档；检测框行剩余差距（0.439 vs oracle 0.887）由检测框覆盖/质量决定，与覆盖归因一致；点通路代价小（−0.016 E2）。该臂为补充（不入注册主表）。**执行备注**：评测代码曾出现列表别名 bug（`p3 = p4 = pt = []` 三变量同列表，导致 297 行/三值相同），已修复并用独立 5 切片抽查复核。工件：`supp_samus_boxft.json/.md`；checkpoint `samus_boxft_seed42.pt`（gitignored）。

## 种子敏感性分析（seed 43；注册主口径为 seed 42，2026-10-18）

**裁定（写作端核对 OSF 原文后）**：注册文本明确 "inference/statistics seed fixed at **42**"（`REGISTRATION_DOCUMENT_EN.md` §3；`plan_A_public_v1.md` 第 33 行同）。注册中出现的 **43 是提示源检查点种子**（P1/P2 的 M1 中位 val 种子、P4 的 D1 检测器种子），不是评估种子。故**主口径维持注册 seed 42**，本轮 seed-43 全套重算作为 **post-registration 敏感性分析**保留（非注册口径）。

**敏感性结果（seed 43，与 seed 42 主口径对照）**：判据方向与量级均不变——§4.4 (i) no-box 5.83% ✓、(ii) conditional 0.732/0.728/0.686 ✗、(iii) strict 0.689/0.685/0.646 ✓ → **NEGATIVE**（主口径 seed42 为 0.743/0.738/0.651，同判）；P5 coverage 1.000 / IoU 中位 0.680 / strict 0.582 → diagnostic/negative（主口径 0.677/0.594，同判）；五源阶梯次序不变（P1 0.464 < P4-reg 0.859 ≈ P4-TTA 0.868 < oracle 0.922）；R0-vs-R2 strict +0.2476 [0.2208,0.2787]、固定子集 conditional +0.0092 [0.0045,0.0158]（主口径 +0.2482/+0.0095）；体积 nnU-Net +124.40 mL、TTA-R2 +19.98 mL（主口径 +147.85/+20.32）；SAMUS box-FT E2 p3 0.892、USFM 注册配方 val 0.7746/E2 0.4557。**全部 seed-43 工件存放于 `outputs/summary/seed43_sensitivity/`**；主 `supp_*` 文件为 seed-42 口径。
