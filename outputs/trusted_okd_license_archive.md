# TRUSTED 数据许可与来源存档

生成时间：2026-10-08（M0 里程碑，自动存档）

## 数据来源（方案 v1.0 中 Dataverse DOI 已失效，实际托管如下）

- **实际托管**：Springer Nature figshare
  - 页面：https://springernature.figshare.com/articles/dataset/TRUSTED_The_Paired_3D_Ultrasound_and_CT_Human_Data_for_Kidney_Segmentation_and_Registration_Research/27981050
  - DOI：`10.6084/m9.figshare.27981050`
  - 整包直链（匿名可用）：`https://springernature.figshare.com/ndownloader/articles/27981050/versions/1`
  - 大小：14.89 GB（2025-04-13 发布）
- 论文（Scientific Data 2025，DOI `10.1038/s41597-025-04467-1`，PMC11993632）Data Records 一节明确引用上述 figshare DOI。
- 方案 v1.0 原写 Dataverse DOI `10.34894/VQ1DJA`：该 DOI 现解析到无关资源（CoreTrustSeal PDF），已废弃。

## 许可结论

- **数据本体（figshare 记录）**：CC BY 4.0（figshare 页面标注）。
- **论文文本**：© Springer Nature Limited 2025，论文本身 CC BY-NC-ND 4.0（仅约束论文，不约束数据）。
- 论文正文使用条款摘录（Europe PMC 全文 XML，2026-10-08 抓取）：
  > "The TRUSTED dataset may be used freely to develop and validate segmentation and IMIR methods."
  > "Because this dataset contains human medical data, before downloading the dataset, users must first agree and email a signed copy of the Data Usage Agreement, as mentioned in the Data Records section."
  > "data_use_agreement.txt, which is a user agreement license to be completed by anyone requesting to download this dataset."

## 待办（正式合规）

- [ ] 数据包内的 `data_use_agreement.txt` 按要求填写并邮件回签给数据作者（论文规定的正式流程；下载为匿名通道，正式使用应以回签 DUA 为准）。**需用户完成**（需真实签名与邮箱）。

## 事实要点（用于校验脚本）

- 48 患者 / 96 肾；CT 48 个 nifti（512×512×586±219 voxels，0.81±0.08 × 0.81±0.08 × 1.05±0.28 mm spacing）
- 目录：`CT_DATA/`（8个子目录）、`US_DATA/`（同构）、`Initial_Transforms_from_noisy_landmarks/`（可选包）、`README.txt`（含官方5折划分：`X_cv_i` = 第 i 折测试集患者ID）、`data_use_agreement.txt`
- 标注：Annotator1 / Annotator2 两套分割 + STAPLE 金标准 + 每肾 7 个解剖标志点；文件名以匿名患者ID为前缀；**每个 3DUS 体只见一个肾**（CT 每例两肾）
- 作者官方基准（论文表）：US 侧 3DUNet/VNet/nnU-Net/CoTr Dice 70.51%–80.70%（本文 E2 外验的文献锚点）
- 医师间一致性 >93% Dice（论文摘要）→ 本方案 M1 门 IoD 期望 0.90–0.96 一致

## OKD 数据许可存档

- 官方获取：Qualtrics 注册（用户已于 2026-10-08 前完成，作者 Dr. Rohit Singla 邮件回复数据链接）
- 数据：经 Qualtrics 注册后由作者邮件发送 OneDrive 分享链接（每位注册者专用，不公开）
- 许可：CC BY-NC-SA（非商用，来自 kidneyUS README）
- 引用要求：Singla, R., Ringstrom, C., Hu, G., Lessoway, V., Reid, J., Nguan, C., & Rohling, R. (2023, October). The open kidney ultrasound data set. In International Workshop on Advances in Simplifying Medical Ultrasound (pp. 155-164). Cham: Springer Nature Switzerland
- 数据事实（README）：>500 张 2D B-mode 腹部超声 PNG（DICOM 不提供）；**每人最多一图**（若校验确认则"划分即患者级"）；两套细粒度多边形标注（VGG Image Annotator CSV，含肾包膜类）；含 native/transplant 肾类型、视角、探头机型等元数据
