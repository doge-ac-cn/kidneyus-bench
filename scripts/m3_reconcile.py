#!/usr/bin/env python3
"""M3 单元完整性对账：核对「模型 × 种子 × 提示 × 数据集」期望矩阵 vs 实际 CSV 行。

期望矩阵（方案 §6/§14 v1.4）：
  E1 (okd test, 99 例) 与 E2 (trusted, 18,322 有效切片)：
    unet   : 3 种子 × prompt=none
    nnunet : 3 种子 × prompt=none（另行打分，此处只查主 CSV；sens 在 *_sens_final）
    medsam : 3 种子 × prompt ∈ {p3, p1, p2, p4}
    sam_lora: 3 种子 × prompt ∈ {p3, p1, p2, p4}
    samus  : 3 种子 × prompt ∈ {p3, p1, p2, p4}（v2 权重）
    usfm   : 3 种子 × prompt=none
  hd95 列：OKD=hd95_px、TRUSTED=hd95_mm；NaN 视为「pred 全空」需单独列出。

用法：python m3_reconcile.py   （可随时重复运行；未完成单元会标 INCOMPLETE）
"""
import csv
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MET = os.path.join(ROOT, "outputs/metrics")

MODELS_PROMPTS = {
    "unet": ["none"],
    "nnunet": ["none"],       # 主CSV行由 m3_score_nnunet --tag best 写入
    "medsam": ["p3", "p1", "p2", "p4"],
    "sam_lora": ["p3", "p1", "p2", "p4"],
    "samus": ["p3", "p1", "p2", "p4"],
    "usfm": ["none"],
}
SEEDS = [42, 43, 44]
DATASETS = {"okd": ("e1_internal.csv", 99, "hd95_px"),
            "trusted": ("e2_external.csv", 18322, "hd95_mm")}


def main():
    problems = []
    for ds, (csv_name, n_cases, hd_col) in DATASETS.items():
        path = os.path.join(MET, csv_name)
        if not os.path.exists(path):
            print(f"[{ds}] {csv_name} 不存在——评测未开始")
            continue
        cells = Counter()
        nan_hd = []
        bad_cols = 0
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("model") == "model":  # 重复表头行（历史遗留，统计时过滤）
                    continue
                key = (row["model"], int(row["seed"]), row["prompt"])
                cells[key] += 1
                if len(row) < 5:
                    bad_cols += 1
                if hd_col in row and row[hd_col].strip().lower() == "nan":
                    nan_hd.append((key, row.get("case_id", "")))
        print(f"\n===== {ds} ({csv_name}, 期望 {n_cases} 例/单元) =====")
        for model, prompts in MODELS_PROMPTS.items():
            for seed in SEEDS:
                for pr in prompts:
                    got = cells.get((model, seed, pr), 0)
                    status = "OK" if got == n_cases else ("INCOMPLETE" if got else "MISSING")
                    if status != "OK":
                        problems.append((ds, model, seed, pr, got, status))
                        print(f"  [{status}] {model} seed{seed} {pr}: {got}/{n_cases}")
        # 正常单元计数汇总
        ok_lines = []
        for model, prompts in MODELS_PROMPTS.items():
            for seed in SEEDS:
                for pr in prompts:
                    got = cells.get((model, seed, pr), 0)
                    if got:
                        ok_lines.append(f"{model}/s{seed}/{pr}={got}")
        print("已落表单元: " + ", ".join(ok_lines) if ok_lines else "已落表单元: 无")
        if nan_hd:
            print(f"  ⚠ pred 全空(NaN HD95)行 {len(nan_hd)} 条——跨域预期行为, 统计时单独计数: {nan_hd[:5]}")
        if bad_cols:
            problems.append((ds, "列数异常行", bad_cols))

    print("\n===== 结论 =====")
    if problems:
        print(f"{len(problems)} 个单元缺失/异常：")
        for p in problems:
            print("  ", p)
        sys.exit(1)
    print("全部期望单元完整，无跳过。")


if __name__ == "__main__":
    main()
