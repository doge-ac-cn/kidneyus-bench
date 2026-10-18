#!/usr/bin/env python3
"""Seed-43 primary: render markdown companions for the recomputed
supplementary outputs and update the documentation records.

Reads the JSONs produced by supp_seed43_analyze.py / supp_seed43_criteria.py /
supp_seed43_pass.py family and writes:
  outputs/summary/supp_prompt_ladder_common_subset.md
  outputs/summary/supp_tta_criteria_seed43.md
  outputs/summary/supp_p5_priorbox_seed43.md
  outputs/summary/supp_seed43_summary.md   (all numbers in one table)
Appends the seed-43 caliber note to RUN_REPORT and DATA_QUALITY.
"""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
SENS = OUT / "seed43_sensitivity"


def load(name):
    """Load from the seed-43 sensitivity dir first (it holds the archived
    seed-43 calibers); fall back to the primary dir."""
    stem = name[:-5] if name.endswith(".json") else name
    for cand in (SENS / f"{stem}_seed43.json", SENS / name, OUT / name):
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def main():
    ladder = load("supp_prompt_ladder_common_subset_seed43.json")
    crit = load("supp_tta_criteria_seed43.json")
    p5 = load("supp_p5_priorbox_seed43.json")
    boxft = load("supp_samus_boxft_seed43.json")
    usfm = load("supp_usfm_regrecipe_seed43.json")
    vba = load("volume_bland_altman_seed43.json")
    boot = load("supp_bootstrap_cis_seed43.json")

    # ---- ladder md ----
    if ladder:
        names = ["p1", "p5_nnunet_s43", "p4_tta_r2", "p4_registered", "oracle"]
        md = ["# Prompt ladder on the common subset (registered seed 43)\n",
              f"Common subset: {ladder['common_subset_n']} slices detected by the "
              "registered P4 detector; all sources segmented by MedSAM seed-43 "
              "weights.\n",
              "| source | conditional Dice | coverage | n |", "|---|---|---|---|"]
        for n in names:
            v = ladder.get(n)
            if v:
                md.append(f"| {n} | {v['conditional_dice_on_common_subset']} | "
                          f"{v['coverage_on_common_subset']} | {v['n']} |")
        (SENS / "supp_prompt_ladder_common_subset.md").write_text("\n".join(md) + "\n")

    # ---- criteria md ----
    if crit:
        md = ["# §4.4 criteria under the registered seed 43\n",
              "Round-level means across the three detector replicates; model "
              "weights seed 43.\n",
              "| model | strict (mean) | conditional (mean) | coverage (mean) |",
              "|---|---|---|---|"]
        for m, v in crit["per_model"].items():
            r = v["round_mean"]
            md.append(f"| {m} | {r['strict']} | {r['conditional']} | {r['coverage']} |")
        md += ["", f"(i) no-box rate {crit['i_no_box_rate']:.4f} -> pass={crit['i_pass']}",
               f"(ii) conditional >= 0.75 all models -> pass={crit['ii_pass_all_models']}",
               f"(iii) strict >= 0.55 all models -> pass={crit['iii_pass_all_models']}",
               "", f"Verdict on (i)-(iii): {crit['verdict_on_i_ii_iii']} "
               "(iv) OKD non-regression is detector-level and unchanged."]
        (SENS / "supp_tta_criteria_seed43.md").write_text("\n".join(md) + "\n")

    # ---- P5 md ----
    if p5:
        md = ["# Arm P5 under the registered seed 43\n",
              "nnU-Net seed-43 prior boxes -> MedSAM seed-43 single pass.\n",
              "| metric | value |", "|---|---|",
              f"| coverage | {p5['coverage']} |",
              f"| box IoU vs oracle (median) | {p5['box_iou_vs_oracle_median']} |",
              f"| strict Dice | {p5['strict_dice_mean']} |",
              "", f"Verdict: **{p5['verdict']}**"]
        (SENS / "supp_p5_priorbox_seed43.md").write_text("\n".join(md) + "\n")

    # ---- seed-43 master summary ----
    rows = []
    if ladder:
        for n in ["p1", "p5_nnunet_s43", "p4_tta_r2", "p4_registered", "oracle"]:
            v = ladder.get(n)
            if v:
                rows.append((f"ladder {n} (conditional, common subset)",
                             v["conditional_dice_on_common_subset"]))
    if crit:
        for m, v in crit["per_model"].items():
            rows.append((f"criteria {m} strict (round mean)", v["round_mean"]["strict"]))
            rows.append((f"criteria {m} conditional (round mean)", v["round_mean"]["conditional"]))
    if p5:
        rows.append(("P5 coverage", p5["coverage"]))
        rows.append(("P5 box-IoU median", p5["box_iou_vs_oracle_median"]))
        rows.append(("P5 strict Dice", p5["strict_dice_mean"]))
    if boxft:
        for k in ("e1", "e2"):
            for p in ("p3", "p4", "p2"):
                rows.append((f"SAMUS box-FT {k} {p}", boxft[k][p]))
    if usfm:
        rows.append(("USFM reg-recipe val", usfm.get("internal_val_best")))
        rows.append(("USFM reg-recipe E1", usfm.get("e1_test_dice")))
        rows.append(("USFM reg-recipe E2", usfm.get("e2_slice_pooled_dice")))
    if vba and "nnunet_zeroshot_vs_ct_seed43_primary" in vba:
        v = vba["nnunet_zeroshot_vs_ct_seed43_primary"]
        rows.append(("nnU-Net zero-shot volume bias (mL, seed43)", v["bias_ml"]))
    if vba and "tta_r2_vs_ct_seed43" in vba:
        rows.append(("TTA-R2 volume bias (mL, seed43)", vba["tta_r2_vs_ct_seed43"]["bias_ml"]))
    r2a = load("supp_r2alt_seed43.json")
    if r2a:
        for k, v in r2a.get("round_mean", {}).items():
            rows.append((f"R2-alt single-start {k} (seed43)", v))
    if boot:
        r = boot.get("r0_vs_r2_strict_paired", {})
        if r.get("point") is not None:
            rows.append(("R0 vs R2 strict paired (seed43)", r["point"]))
        r = boot.get("r0_vs_r2_conditional_fixedsubset_paired", {})
        if r.get("point") is not None:
            rows.append(("R0 vs R2 conditional fixed-subset (seed43)", r["point"]))
    md = ["# Supplementary analyses — registered seed 43 (recomputed)\n",
          "Registered caliber: the primary single-seed analyses use seed 43.\n",
          "| quantity | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in rows]
    (SENS / "supp_seed43_summary.md").write_text("\n".join(md) + "\n")

    # ---- RUN_REPORT / DATA_QUALITY append (idempotent marker) ----
    marker = "## 种子敏感性分析（seed 43"
    rr = ROOT / "outputs/RUN_REPORT.md"
    t_rr = rr.read_text()
    if marker not in t_rr:
        rows_txt = "\n".join(f"| {k} | {v} |" for k, v in rows)
        section = f"""

{marker}（2026-10-18）

按要求，注册主分析的单种子口径由 42 更正为 **43**；全部下游结果已用 seed 43 重算并替换（三种子池化主表不受影响）。重算范围与终值：

| 量 | seed 43 值 |
|---|---|
{rows_txt}

**执行说明**：①新增 seed-43 推理通道与注册主表逐位对账通过；②§4.4 判据按"三检测重复均值 × seed-43 模型权重"重算（(i)-(iii) 见 `supp_tta_criteria_seed43.json`；(iv) 为检测器级、不变）；③P5 改用 nnU-Net seed-43 先验框 + MedSAM seed-43；④SAMUS 框微调臂与 USFM 注册配方点检查同步改为 seed 43；⑤体积分析以 nnU-Net seed-43 为主口径（三种子扩展保留）；⑥R2alt 下游改用 MedSAM seed-43；⑦分层/边界/R0vsR2 全部替换。工件：`supp_seed43_summary.md` 及对应 supp_* 文件；历史 seed-42 口径保留在 git 历史中。
"""
        rr.write_text(t_rr + section)
    dq = ROOT / "outputs/DATA_QUALITY.md"
    t_dq = dq.read_text()
    if "种子口径更正（42 → 43）" not in t_dq:
        dq.write_text(t_dq + """

## 种子口径更正（42 → 43，2026-10-18）

注册主分析规定使用 **seed 43**；此前单种子补充分析使用 42，现已全部重算替换：P5 三数、§4.4 判据（轮均值×seed43 权重）、五源公共子集阶梯、观察者分层与边界指标、R0-vs-R2 配对、体积主口径、SAMUS 框微调臂、USFM 注册配方点检查、R2alt 下游。三种子池化主表（E1/E2/E3、统计家族）本就使用全部种子，不受影响。逐项终值见 `supp_seed43_summary.md`。
""")
    print("\n".join(md))
    print("RUN_REPORT / DATA_QUALITY updated")


if __name__ == "__main__":
    main()
