# 方案A执行文档：超声肾脏分割基础模型跨库基准（零人工标注版 v1.0）

> 生成日期：2026-10-08 ｜ 性质：可无人值守执行的服务器实验规划
> 目标产出：一篇 **JCR 三区（Q3）及以上**的医学影像AI论文（超声肾脏分割·基础模型系统评测与跨库外部验证）的全部实验结果。中科院分区不作为约束；本方案实验深度按 JCR Q2 标准设计，Q3 为保底线。
> **铁律：全程零人工标注。** 人工box消融已删除，替换为全自动提示策略对比；所有"人类参考值"一律来自数据集自带的医师标注，脚本只做读取与计算。

---

## 0. 项目一句话

在 Open Kidney Dataset（OKD，514张2D B超，自带分割标注）上训练/适配5个分割模型，在 TRUSTED（48患者/96肾，配对3D超声+CT，自带双医师标注与STAPLE共识）上整库外部测试，输出：内部/外部性能、跨库退化率、以医师一致性为上界的归一化性能、自动提示策略消融、自体vs移植肾亚组、3D体积级聚合结果。

## 1. 人力需求声明（给执行方）

- 医学标注：**0小时**（不存在任何需要医学背景的环节）
- 工程人力：数据下载校验 + 跑脚本 + 按里程碑检查日志，预计每周2–4小时看护
- 硬件：单张24GB GPU（RTX 4090/3090/A5000均可），磁盘≥500GB，总GPU预算约250–300小时

## 2. 环境与依赖

- CUDA ≥ 12.1，PyTorch ≥ 2.3，Python 3.10
- 核心包：`nnunetv2`（MIC-DKFZ/nnUNet）、`MedSAM`（bowang-lab/MedSAM）、`segment-anything`（facebookresearch/segment-anything）、`SAM-2`（facebookresearch/sam2）、`SimpleITK`、`MONAI`、`pydicom`、`scipy`、`statsmodels`、`medpy`（HD95）、`staple`（若需自行STAPLE融合；优先用TRUSTED自带共识）
- MedSAM与SAM官方预训练权重需提前下载；SAM-MedUS（Tian等2025）权重在GitHub/HuggingFace检索"SAM-MedUS"，**若2026-11-01前仍无法获得可用权重 → 直接放弃该模型并在报告中注明**，不阻塞主线
- 环境快照：`pip freeze > requirements.lock`，随机种子固定脚本 `set_seed(42)` 贯穿所有训练/划分/初始化

## 3. 数据获取与M0验收（第1周，go/no-go检查点）

| 数据 | 获取方式 | 用途 |
|---|---|---|
| OKD | Zenodo DOI `10.5281/zenodo.5983586`（CC-BY）；配套 github.com/rsingla92/kidneyUS | 训练+内部测试 |
| TRUSTED | 数据DOI `10.34894/VQ1DJA`（Dataverse）；预印本 hal.science/hal-05041836 | 仅作外部测试+3D实验 |

**M0验收门（全部通过才进入训练）：**
1. OKD：图像数=514，mask可解析为二值标签；**查找患者ID字段**——若每人一图则在报告中注明"划分即患者级"；若一人多图，按患者ID做患者级划分
2. TRUSTED：患者数=48、肾脏数=96；核对标注内容应含：医师A标注、医师B标注、STAPLE共识（若缺STAPLE则用 `sitk.MultiLabelSTAPLE` 自行融合并注明）、7个解剖标志点
3. 许可确认：TRUSTED数据存储页许可截图存档（论文PDF为CC BY-NC-ND，数据本体声明可自由使用——以存储页为准）
4. 完整性：3D超声体数据能以SimpleITK读出，记录各体 spacing/尺寸分布
5. Sanity check：MedSAM零样本（auto-box由图像质心生成）在OKD随机20张上推理一次，确认环境跑通并记录Dice

## 4. 数据划分与预处理（全部脚本化，划分清单随论文开源）

- **内部库 OKD**：60/20/20 训练/验证/测试；患者级不交叉；分层字段不可得（见§10规则3），按 view 类型分层；种子42生成**唯一**canonical划分并导出 `splits.json`（v1.4/T0-5：划分清单含数据集内部匿名编号，依隐私规则不入公开仓库——**凭注册获得的同一数据集 + 种子42 + scripts/m1_split_okd.py 可精确复现**，论文措辞用"seed-reproducible"而非"released"）
- **外部库 TRUSTED**：整库96肾只做测试，任何训练不可见
- **2D切片抽取规则（TRUSTED 3D→2D）**：对每个3D US体，取共识mask面积≥50像素的层面为有效切片；记录层面索引；同一体内切片不跨训练
- **预处理统一**：重采样至各模型要求（nnU-Net内部自理）；灰度窗口取1–99百分位后归一化；mask最近邻插值；所有预处理参数写入 `preprocess_config.json`
- **金标准**：外部测试用STAPLE共识；医师A/B各自标注用于计算人体一致性上界IoD（v1.2：主口径为**直接A-vs-B**，A/B-vs-共识的自洽口径并列报告）
- **OKD训练/评测GT来源（v1.2写死）**：标注者1（`reviewed_masks_1/capsule`），与官方锚点 annotator_1 变体（nnU-Net v1 五折CV 0.8852）同口径；所有模型（M1–M5、D1标签推导）统一该来源，禁止中途换靶重训
- **排除规则佐证（v1.2）**：48例空mask中47例带数据集自带 Poor/Unsatisfactory 质量标记、46例两位标注者一致未标——队列排除与数据集作者自身的质量把关一致，非本方案自由裁量（论文表1血缘链引用）

## 5. 模型清单（5个，全部公开权重/公开实现）

| # | 模型 | 类型 | 输入与提示 | 训练 |
|---|---|---|---|---|
| M1 | 轻量U-Net | 从零对照 | 2D灰度 | OKD训练，Dice+BCE损失，AdamW lr=1e-3，cosine，100ep |
| M2 | nnU-Net v2（2D配置） | 任务专用SOTA | 2D | OKD训练，nnU-Net默认超参（不改动） |
| M3 | MedSAM | 医学基础模型 | 2D + box提示（见§6提示策略） | OKD微调（全参，lr=1e-5，50ep） |
| M4 | SAM + LoRA（vision encoder r=8） | 通用基础模型适配 | 2D + point/box提示 | OKD微调分割头+LoRA，lr=1e-4 |
| M5 | SAM-MedUS | US专用基础模型 | 2D + box/point提示 | 官方权重直接推理 + OKD轻微微调两版 |
| D1 | YOLOv8s/YOLO11s（检测组件，非分割模型） | 目标检测器 | 2D灰度转3通道 | **训练标签由OKD现成mask自动转外接矩形生成（零人工标注）**；Ultralytics默认超参，50ep，3种子；若Ultralytics不可用则fallback torchvision Faster R-CNN |

> D1的定位：为M3/M4/M5提供**全自动提示框**，构成"检测→分割"端到端无人工管线；同时自身输出检测精度表。

- 每个可训练模型跑 **3个训练种子（42/43/44）**，结果报 mean±sd；推理与统计种子固定42
- 所有模型在同一预处理、同一划分、同一评价脚本上出数，禁止各模型私自后处理（3D聚合后处理除外，统一规则见§6-E4）

## 6. 实验矩阵

**E1 内部基准（OKD测试集）**：5模型×3种子 → Dice/HD95，患者级Bootstrap 95%CI。
**E1+ 文献锚点（v1.2新增）**：nnU-Net v2 **单种子全量5折CV**（486例、患者级分折——比官方的随机分折更严格），作为与官方0.8852（v1+CV口径）同协议对话的锚点；仅nnU-Net跑CV（用途是文献绝对数对照，模型排名仍以共同固定测试集为准，论文写明该分工）；排在M3之后、M5之前执行，约55 GPU时。
**E1++ 分辨率敏感性（v1.3可选加强①，采纳）**：U-Net 以512×512重训×3种子（约10 GPU时），验证E1模型排序不是输入分辨率伪影（U-Net 256 vs 基础模型1024）；与256结论一致则256为主表、512入附录。

**E2 跨库外部验证（TRUSTED全库）**：5模型在OKD训练→TRUSTED 2D切片直接推理；计算 Dice、HD95、跨库退化率 `Δ=(Dice内−Dice外)/Dice内`、归一化Dice `Dice/IoD`。

**E3 提示策略消融（替代人工box，全自动，M3/M4/M5适用）**：
- **P1 先验框**：用M1（U-Net）在OKD上训练的粗分割预测取最大连通域外接矩形作为box/质心作为point——完全不碰真值
- **P1执行细则（v1.2）**：先验源固定为 M1 三个种子中**val Dice中位数**的单一checkpoint（当前=seed43/0.6975；seed44后期发散、val=0不得用作先验源），禁止best-of挑选；D1框或先验为空/失效的病例，该例Dice记0并计入分母，禁止静默剔除；E3表并列报告P1/P4提示框相对P3框的IoU，使"提示源退化"与"提示类型差异"可归因
- **P2 质心点**：同一先验的质心单点提示
- **P4 检测框（端到端全自动，正式部署管线）**：D1（YOLO）先在OKD上训练（bbox由现成mask自动推导），再对OKD测试集与TRUSTED整库直接检测出框，喂给M3/M4/M5做提示——推理阶段**完全不依赖分割先验和真值**，是"检测→基础模型分割"全自动管线的正式成绩
- **P3 Oracle框（诊断性上界，非部署场景）**：由数据集自带mask自动计算的外接框——**这是从现成标注脚本化推导，无任何人工作业**；论文中明确标注为上界参照
- 对比维度：P4 vs P1 vs P2 = 三种全自动提示策略排序；P_x vs P3 差距 = 提示敏感性/距上界距离；P4作为主推部署方案
- **检测器自身指标**：D1在OKD测试集与TRUSTED上的 bbox IoU/mAP@0.5（参考框由共识mask自动推导），单独成表——同时论证"检测环节本身在跨库下掉多少"

**E4 3D体积级、反向实验与退化归因**：
- TRUSTED每个US体的2D预测切片 → 逐层堆叠 → 最大连通域+形态学闭运算 → 体积级Dice/体积误差%（统一后处理规则，写死参数）
- TRUSTED的CT侧：用同一批模型直接推理（OKD训练、CT域），仅作**跨模态退化诊断**（预期大幅退化，如实报告即可，不做调参）
- 反向3D上界：nnU-Net v2 3D配置在TRUSTED上5折交叉验证（官方划分，训练不可见留出折），量化"任务专用3D"性能上界
- **反向2D对照（v1.1新增）**：nnU-Net v2 **2D配置**在TRUSTED上与反向3D**同折同数据**5折交叉验证。与E2构成退化归因矩阵：E2（OKD训练→TRUSTED）对比反向2D = **域退化**（跨库泛化损失）；反向2D对比反向3D = **格式损失**（2D/3D错配）——直接回应"2D训练→3D测试归因混淆"的审稿质疑；层内（切片级）分析已在E2单独报告
- **临床落点（v1.1新增）**：体积Bland-Altman以mL报告95%一致性区间（LOA），并对照文献临床阈值（肾体积测量重复性系数、TKV随访变化>5%等）判读"Dice差异是否具有临床意义"——回应"so-what"质疑
- 预算注记：反向2D约+60–85 GPU时（约1–1.5min/epoch×1000ep×5折），3卡并行1.5–2天；GPU紧张时按§10规则6降至3折并注明

**E6 部署指标（v1.3可选加强③，采纳）**：在E2推理时顺带记录各模型单切片推理时延（ms）与峰值显存（MiB，V100-16GB、FP16），形成部署成本表——支撑P4"检测→分割"全自动管线的床旁可行性叙事，边际成本≈0。

**E5 亚组与失败分析（全自动，无人工阅片）**：
- 自体 vs 移植肾（OKD元数据分层；若元数据无此字段，用文件名/CSV可恢复则用，否则**删除该分析**并在报告注明）
- 失败相关因素回归：Dice vs 图像对比度/SNR代理指标、共识mask面积、mask质心深度（行索引）——全部可自动计算
- 定性图：按Dice五分位自动抽取 代表性成功/失败病例 各5例出图（论文选图时再由作者目检，非实验环节）

## 7. 评价指标与统计（一次写死）

- 逐切片：Dice、HD95(mm)；逐体积（TRUSTED）：Dice、体积误差%、HD95
- 人体参照：IoD=医师A/B标注对共识的Dice均值；归一化Dice=模型Dice/IoD。（v1.2）因共识由A/B导出，A/B-vs-共识为自洽偏乐观口径；**主口径改用直接A-vs-B**，自洽口径并列——归一化Dice两口径都报
- 统计：同一测试集多模型比较 Friedman + Holm校正；两两比较 Wilcoxon符号秩；AUC类（不适用本方案）；全部指标患者级Bootstrap（n=1000）95%CI；α=0.05；统计脚本输出 `stats_report.csv`（含每对的p值、效应量、CI）
- **3种子的统计处理（v1.2写死，防伪重复）**：逐患者对3个种子取均值 → 每模型得到一条患者级向量 → Friedman/Wilcoxon/Bootstrap 作用于种子均值向量；**严禁 3种子×患者 展开为独立观测入检验**；种子间sd仅作训练稳定性描述，不进入任何检验自由度；辅助报告每个预声明配对的"3/3种子方向一致率"
- **Bootstrap聚类单位（v1.2写死）**：OKD按患者（93簇，双图患者先取均值）；TRUSTED按患者（39簇，携带双肾与全部切片整体重采样）——不得按肾（59）或切片（18322）重采样
- **体级聚合顺序（v1.2写死）**：体切片数637–868不等，先肾内聚合、再跨59肾宏平均，禁止逐切片池化
- **双口径CI并列（v1.3可选加强⑥，采纳）**：TRUSTED同时报告患者级（39簇）与肾级（59簇）两种聚类的95%CI，展示结论对聚类单位不敏感
- **混合模型互证（v1.3可选加强②，采纳）**：线性混合模型（Dice ~ 模型固定效应 + 患者随机效应 + 种子随机效应）与Friedman结论互证，结果入附录
- 必须检验的四组对比：M3(MedSAM) vs M2(nnU-Net)；内库 vs 外库（各模型配对）；P4(检测框) vs P1(先验框) vs P2(点提示)（全自动策略排序）；P_x vs P3（提示敏感性）
- **功效局限（v1.2预声明）**：TRUSTED仅39个患者簇，<2个Dice点的差异基本不可检出；CI重叠时只许表述"未检出差异"，禁止表述"性能相当"；主结论不依赖TRUSTED上的显著性星号

## 8. 输出规范（服务器按此落盘，缺一不可）

```
outputs/
├── RUN_REPORT.md            # 自动生成：各里程碑通过状态、环境快照、数据校验结果
├── splits.json              # canonical划分（开源件）
├── metrics/
│   ├── e1_internal.csv      # model,seed,case_id,dice,hd95
│   ├── e2_external.csv      # model,seed,patient,kidney_id,slice_idx,dice,hd95
│   ├── e2_external_3d.csv   # 体积级
│   ├── e3_prompts.csv       # model,prompt(P1/P2/P4/P3),dice
│   ├── e3_detection.csv     # D1检测器:dataset,bbox_iou,map@0.5
│   ├── e5_subgroup.csv      # native/transplant
│   └── iod_reference.csv    # 医师A/B对共识Dice（人体上界）
├── predictions/             # NIfTI(TRUSTED 3D)/PNG(OKD 2D)，命名 {model}_{seed}_{case}
├── figures/                 # 箱线图、退化率条形图、Dice-IoD散点、Bland-Altman、失败案例图（matplotlib，300dpi）
└── logs/                    # 训练日志按 model_seed.log 命名
```

## 9. 里程碑与自动验收门

| 里程碑 | 时间 | 内容 | 验收门（不达标自动停下并写RUN_REPORT） |
|---|---|---|---|
| M0 | 第1周 | 数据下载、校验（§3） | 48/96、514、许可存档、sanity跑通 |
| M1 | 第2–3周 | 预处理+划分+IoD计算 | IoD落在0.90–0.96（与TRUSTED论文>93%相符，否则查标注解析） |
| M2 | 第3–8周 | D1检测器训练 + 5模型×3种子内部训练 | D1在OKD **val** bbox mAP@0.5≥0.9（肾为单类大目标，低于此值查bbox推导），M2(nnU-Net) OKD **val** Dice≥0.80，否则排查管线（v1.2：验收一律用val，**禁止以任何理由中途读取test**；test评测仅限M3起的正式出数，每次触碰test记入RUN_REPORT访问日志）。官方测试期对照锚点：nnU-Net v1 五折CV 0.8852（口径见§4/§6-E1+，不作为本方案验收阈值） |
| M3 | 第8–11周 | 外部验证+E3提示消融+E5 | 全部96肾×5模型×3种子出数齐全 |
| M4 | 第11–13周 | 3D聚合+CT侧诊断+反向3D | 反向3D 5折完成 |
| M5 | 第13–16周 | 统计+图表+RUN_REPORT定稿 | 三组必检对比均有p值与CI |

预算：总计约260–320 GPU时（M2约130h含YOLO训练<5h、M4约50h、推理与消融约80h），单卡3–4周机时；磁盘峰值<300GB。

## 10. 风险与自动决策规则（无需询问人）

1. **SAM-MedUS权重不可得** → 删除该行，5模型变4模型，报告注明（不阻塞）
2. **TRUSTED缺STAPLE共识** → `sitk.MultiLabelSTAPLE` 自行融合，报告中注明与论文引用数字核对
3. **OKD无患者ID/无自体移植字段** → 划分退化为图像级（报告注明局限）/删除E5亚组
4. **外部Dice异常高（>IoD）** → 立即查数据泄漏（TRUSTED是否混入训练），这是红线检查
5. **TRUSTED 3D某体数据损坏** → 跳过并列入 `excluded_cases.log`，有效样本数写入所有表格脚注
6. **GPU时超预算1.5倍** → 优先砍M4反向3D至3折，其余不动

## 11. 论文图表对应（实验产出直接映射）

表：数据集与划分；主结果（内/外库Dice、HD95、归一化Dice）；提示策略消融（P1/P2/P4/P3，含D1检测精度子表）；亚组与失败分析。
图：流程图（含"YOLO检测→基础模型分割"全自动管线示意）；模型×数据集Dice箱线图；跨库退化率条形图；**退化归因分解图（域退化 vs 格式损失，v1.1新增）**；Dice-IoD散点（离人还有多远）；TRUSTED体积Bland-Altman（95% LOA，mL）；**CV折间方差对照图（v1.3②：v2五折 vs 官方v1五折0.852–0.933 并排）**；提示策略P1/P2/P4/P3对比；D1检测框可视化示例；成功/失败案例图。表：另含推理时延/显存部署成本表（E6）。TRUSTED CT侧跨模态诊断结果放**附录**并配跨模态讨论段（v1.3⑤，白赚半页）。

## 12. 执行顺序速览（runbook）

```
m0_download_and_verify.sh   # 下载、DOI校验、许可存档、格式解析、sanity推理
m1_preprocess_and_split.py  # 预处理、splits.json、IoD
m2_train_internal.sh        # D1检测器训练 + 5模型×3种子（nnU-Net用其CLI，其余单脚本）
m3_external_eval.py         # TRUSTED推理、E3提示消融（P1/P2/P4/P3，D1对TRUSTED直接检测出框）、E5亚组
m4_3d_and_reverse.sh        # 3D聚合、CT侧诊断、TRUSTED 5折 3D+2D nnU-Net（退化归因矩阵）
m5_stats_and_figures.py     # 统计检验、全部图表、RUN_REPORT.md
```

各里程碑通过验收门后自动进入下一个；任一门未过 → 停止并写明原因到 RUN_REPORT.md。

## 13. 投稿期刊阶梯（JCR口径，分区与IF随年度浮动，投稿前以最新JCR核实）

| 梯队 | 期刊 | 口径 | 定位 |
|---|---|---|---|
| 冲刺（JCR Q1/Q2） | Computers in Biology and Medicine | 综合医学AI，Q1 | 结果强（外验退化小+提示消融干净）时投 |
| 冲刺（JCR Q1/Q2） | Computerized Medical Imaging and Graphics / Computer Methods and Programs in Biomedicine | 医学图像计算，Q1/Q2 | 强调方法学与可复现套件时投 |
| 冲刺（JCR Q2） | Journal of Digital Imaging | 影像信息学，Q2 | 强调部署管线（免提示P1）时投 |
| **主投（JCR Q2/Q3，超声专科对口）** | **Ultrasound in Medicine and Biology** | 超声专科，Q2/Q3 | **本方案最对口的期刊**：超声读者群+基准评测体裁友好 |
| 主投（JCR Q2/Q3） | Quantitative Imaging in Medicine and Surgery (QIMS) | 影像，Q2/Q3 | 体裁灵活、周期快 |
| 主投（JCR Q2/Q3） | European Journal of Radiology | 影像，Q1/Q2 | 需在Discussion强化临床落点（肾脏体积测量/移植随访） |
| 保底（JCR Q3） | BMC Medical Imaging | 影像，Q3 | 审稿快、对"严谨评测+公开数据"体裁接受度最高 |
| 保底（JCR Q3） | Journal of Medical Imaging (SPIE) / Int J CARS | 影像，Q3 | 方法细节可完整展开 |

**投稿策略（v1.1按三刊对标分析修订）**：
- **情景A（外验强，归一化Dice≥75%）**：IEEE JBHI（冲刺主选）→ Ultrasound in Medicine and Biology → QIMS 阶梯，不在低IF刊消耗强结果；（v1.4勘误：Computers in Biology and Medicine 已于2025-11被WoS除名，从阶梯移除）；
- **情景B（外验退化大）**：全文改写为"泛化警示基准+失败模式分析"叙事，**Journal of Clinical Ultrasound 最对口**（其读者正需要泛化警示），保底 BMC Medical Imaging / SPIE JMI 不变；
- **JUM（Journal of Ultrasound in Medicine）确认不投**：读者错配（临床医师导向）+约三成初筛拒风险，收益/风险比最差；
- 定位话术统一为"**首个跨库基础模型超声肾脏评测协议+失败模式分析**"（标题/摘要/cover letter），前置防御"benchmark-only"质疑。禁止为分区数字投有口碑风险的掠夺性/水刊，JCR Q3是下限而非目标。

**JCR口径对实验内容的额外要求（相对原规划的两处补强）**：
1. **文献锚定**：E2/E3结果表中须并列引用 UltraBench 2（arXiv 2026-09）与 SAM-MedUS 已发表的公开成绩（含OKD任务），做成"本文 vs 文献"对照表——JCR Q2审稿人必查与既有基准的关系；
2. **可复现包**：代码+划分清单+预测样例随投稿上传（GitHub/Zenodo DOI），JCR Q1/Q2评测类论文缺开源包会直接被压分。

---

## 14. 变更记录

**v1.1（2026-10-09）｜触发：三刊对标分析（UMB/JUM/JCU 近两年30篇排位评审，导师视角）**

1. **E4 新增"反向2D对照"**：nnU-Net v2 2D 配置在 TRUSTED 官方5折上与反向3D同折同数据训练，与 E2 构成退化归因矩阵（域退化 vs 格式损失分离）——直接回应三位评审共同点名的"2D训练→3D测试归因混淆"。约+60–85 GPU时。
2. **E4 新增临床落点**：体积 Bland-Altman 95% LOA（mL）对照临床阈值判读——回应"缺 so-what"质疑（TRUSTED 有 CT 配对，边际成本极低）。
3. **§13 投稿策略情景化**：A线（外验强）CBM/JBHI→UMB→QIMS；B线（外验差）JCU 泛化警示叙事；**JUM 确认排除**；保底 BMC MI / SPIE JMI 不变。
4. **定位话术**统一为"首个跨库基础模型超声肾脏评测协议+失败模式分析"，前置防御 benchmark-only 质疑（写作层，无实验变更）。
5. **不变项**：E1–E3 主线、E3 提示消融、§7 统计方案、零人工标注铁律、可复现包要求——对标分析确认这些正是三刊近两年论文的稀缺项（无基础模型原创应用、几乎无外验、无开源、无规范统计），是本方案的差异化优势，保持不动。
6. **勘误**：对标文档中"48例自体vs移植比较预先声明为探索性"为过时信息——OKD 分发数据无该字段，E5 亚组分析已按 §10 规则3 于 2026-10-08 M0 验收时删除，论文中不得承诺该分析。
7. **对表说明**：TRUSTED 官方基准（70.51–80.70%）为 3D/病人级口径，本方案 E2 为 2D 切片级口径，论文对比时须注明口径差异（M4 反向实验补齐同口径对照）。
8. **v1.2（2026-10-09）｜触发：导师视角方法学评审（Q1/Q2专项+12项审查）**

1. **Q1裁决**：固定患者级划分+3种子作为多模型配对协议站得住（且患者级+dHash审计比官方随机分折更干净），但不足以与官方0.8852（CV口径）对话→ **新增E1+**：nnU-Net v2 单种子全量5折CV（患者级分折）作文献锚点，约55 GPU时，M3后M5前执行；仅nnU-Net享受CV（对照锚点用途），排名仍以共同固定测试集为准。
2. **Q2裁决**：3种子必要且优于官方单种子（M1 seed44 0.5655 vs seed42 0.7092 证明单种子是撞大运）；**统计处理写死进§7**——逐患者种子均值入检验、严禁种子池化、sd仅描述、3/3方向一致率作稳健性；若核心配对的种子间sd>模型间差距1/3，仅对该配对补种子45/46。
3. **M2验收门纠偏（纪律）**：原"OKD测试Dice≥0.80"属训练中途偷看测试集→改为**val阈值**；建立**测试集访问日志**制度（任何触碰test的命令记入RUN_REPORT）。
4. **OKD GT来源写死（§4）**：标注者1 masks，与官方锚点 annotator_1 变体同口径（已核实全部训练代码在用，无需重训）；禁止中途换靶。
5. **IoD双口径（§4/§7）**：直接A-vs-B为主口径（共识由A/B导出，自洽口径偏乐观），两口径并列报告。
6. **E3细则（§6）**：P1先验源=val中位数种子（防seed-shopping）；空框/空先验计Dice=0入分母；提示框相对P3框的IoU进表。
7. **Bootstrap聚类单位写死（§7）**：OKD 93患者簇、TRUSTED 39患者簇（携带双肾+切片整体重采样）。
8. **体级聚合顺序写死（§7）**：肾内聚合→跨59肾宏平均，禁止逐切片池化。
9. **功效局限预声明（§7）**：TRUSTED 39簇下<2点差异不可检出；CI重叠只许说"未检出差异"。
10. **排除规则佐证（§4）**：48空mask中47例自带Poor/Unsatisfactory标记、46例双标注者一致未标——排除有据，写入论文表1血缘链。
11. **U-Net seed44定性**：后期发散（ep100 loss停在0.47、val_dice=0），best为中途值；E1如实报3种子，论文以"训练不稳定性"呈现；该种子禁作P1先验源。
12. **待办（写作层）**：口径三角表（v1/v2×划分×队列规模）；"零新增标注"措辞精确化；投稿前正式查新限定"首个"主张；SAMUS若入列则主表预指定微调版；E4反向实验与E2同脚本同聚合的协议声明。

9. **v1.3（2026-10-09）｜导师评审"可选加强"6项全部采纳**
   ① E1++ 分辨率敏感性：U-Net 512×3种子（~10 GPU时），验证排序非分辨率伪影；② §7 线性混合模型（患者+种子随机效应）与Friedman互证；③ 新增E6 部署指标（单切片时延+峰值显存表，随E2推理顺带完成）；④ §11 CV折间方差对照图（v2五折 vs 官方v1五折并排）；⑤ CT侧诊断移附录+跨模态讨论段（纯写作）；⑥ §7 TRUSTED双口径CI（39患者簇/59肾簇）并列。合计新增GPU时约10（其余为零/顺带），M5统计脚本增加LMM与双口径CI。

**v1.4（2026-10-10）｜触发：导师视角外部评审（方法学/文献对标/投稿推演三智能体）**

**T0 工件一致性（已执行）**：
1. requirements.lock 重生成（此前含坏版本 torch==2.14.1，现 2.7.1+cu126 与环境事实一致）；
2. GitHub 工件补同步：m0_sanity_medsam.json（0.7808 修复版）、iod_reference.csv（双口径列）此前未推送；
3. README plan 链接死链修复；m0_verify_trusted.py 落盘前清洗肾ID（T0-6）；
4. T0-5 splits.json：**维持不入库**（含数据集内部匿名编号，依隐私规则），协议措辞改为 seed-reproducible（§4 已改）——如作者后续决定开源可再翻案；
5. 制度：每里程碑验收后 24h 内推送工件。

**T1 公平性口径（已核对/已执行）**：
1. **T1-1 MedSAM 种子协议：前提不成立，无需重跑**——实证三个种子全部 bs=1（OOM修复补丁在队列启动前已生效，逐日志核对 bs=1），协议本就统一；记录结案；
2. **T1-2 nnU-Net checkpoint**：采纳 best 为主口径（与其余模型 best-val 语义一致）；final 行降级入敏感性文件（e1/e2_internal_sens_final.csv / e2_external_sens_final.csv）——score 脚本已加 --chk/--tag 路由；
3. **T1-3 GradScaler**：三训练脚本已加（unet/foundation；samus 因 v2 链运行中待链后补）；已完成训练不重训，limitation 注明；
4. **T1-4 增广敏感性臂**：--aug 已实现于 unet/foundation 脚本（固定配置：h/v翻转+rot90+亮度±20%/对比度±20%，p=0.5），3种子×2模型，判决规则预注册（增广臂优≥1.5 Dice → 换主表）；排在 GPU 空闲档；
5. **T1-5 主表口径**：headline 改为 **P4（检测框全自动部署口径）**，P3 降为上界参照行——M5 统计按此出主结论。

**T2 新增实验**：
1. **A1 USFM 第六模型（M6）——采纳但修正两处事实**：(a) USFM 为**内部学习式提示**（非用户可交互框/点提示），故 M6 只进 E1/E2（prompt-free，同 unet/nnunet），**不进 E3 消融**（原文"四种提示都要跑"不成立）；(b) 权重仅 Google Drive（USFM_latest.pth）且仓库无 LICENSE 文件 → 需用户从外部网络下载（同 SAMUS 流程）+ 邮件作者要书面许可；不可得则比照 URFM fallback / limitation。微调协议对齐 M3（lr=1e-5、50ep、3种子），进 6 模型统计家族（此刻预声明）；验收门 val Dice ≥0.85；
2. **A2 第三方外验第二队列（UT Kidney，570图/281患者）——采纳**：用户动作=Google表单申请（本机无法访问Google）+ 邮件作者取书面许可（仓库License栏为空，必须有）；数据到位后流程=备忘录（mask口径/厂商人群/机构全称）→ dHash 与 OKD 近重复审计 → E2 同脚本只推理（281患者簇Bootstrap）→ a2_utkidney.csv；UUSIC 2025 因肾脏子集系 OKD 重分发排除 ✓；
3. A3 计划内确认：E1+ CV / E4 反向 / E6 时延 / E1++ lr3e-4 重跑——按 runbook 原序，无新增。

**排产护栏**：T0 立即✅；T1-1 结案；T1-2 的 best 重预测排 GPU0 空闲档（v2 链后自动）；T1-4 增广臂与 A1 USFM 排 GPU 空闲档；A2 等用户申请数据。**M3 外验数字落盘 = 实验冻结，此后只做统计与写作。**统计家族扩为 6 模型（此刻预声明），headline=P4。

**预注册生效（2026-10-10）**：OSF 注册 DOI [10.17605/OSF.IO/TX7RN](https://doi.org/10.17605/OSF.IO/TX7RN)，注册内容为方案整合版（REGISTRATION_DOCUMENT_EN.md + plan_A_public_v1.md）。自本时刻起实验冻结：仅允许已排队任务收尾（USFM、增广臂、nnU-Net best 重评分、SAMUS v2、E1+ CV、E4 反向、UT Kidney 推理）；任何偏离写入论文 Post-registration Deviations 小节；测试集访问日志与泄漏红线照旧。论文 Methods 引用模板：*The study protocol was registered on the Open Science Framework prior to the aggregation and reporting of external-validation results (https://doi.org/10.17605/OSF.IO/TX7RN).*
