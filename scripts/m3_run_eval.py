#!/usr/bin/env python3
"""M3 main evaluation: models x seeds x datasets x prompts -> metric CSVs.

Protocol:
  - E1/E2 headline prompt for prompted models = P3 (oracle box, plan §6);
    prompt-free models (unet/nnunet) run prompt-free and are tagged "none"
  - E3 ablation adds P1/P2/P4 from outputs/prompts/*.npz (stage 1)
  - empty/missing prompt -> Dice 0, HD95 NaN (v1.2; counted by stats stage)
  - all predictions resampled to the native GT grid; one shared scorer
CSVs (append-safe): outputs/metrics/{e1_internal,e2_external,e3_prompts}.csv
"""
import argparse
import os
import sys

import re

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as M  # noqa: E402
from utils_kidney import load_json  # noqa: E402

METRICS = os.path.join(ROOT, "outputs/metrics")


def append_row(csv_name, header, row):
    os.makedirs(METRICS, exist_ok=True)
    path = os.path.join(METRICS, csv_name)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")


def get_prompt(case_id, prompt, dataset, gt, cache):
    """Returns (box_native, point_native, tag)."""
    if prompt == "p3":
        return M.oracle_box(gt), None, "p3"
    code = prompt.split("_")[0]  # accept both "p1" and "p1_box"
    kind = {"p1": "p1_box", "p2": "p2_point", "p4": "p4_box"}[code]
    v = cache.get(kind, {}).get(case_id)
    if code == "p1":
        return v, None, "p1"
    if code == "p2":
        return None, v, "p2"
    if code == "p4":
        return v, None, "p4"
    raise ValueError(prompt)


@torch.no_grad()
def infer_prompted(model_name, model, img, prompt_kind, box, point, device):
    if model_name in ("medsam", "sam_lora"):
        res = M.MODEL_RES[model_name]
        return M.infer_sam_family(model, img, res, device, box=box, point=point, bs=1)
    if model_name == "samus":
        return M.infer_samus_box(model, img, device, box=box, point=point)
    raise ValueError(model_name)


@torch.no_grad()
def infer_unet_case(model, img, device):
    return M.infer_unet(model, img, device)


def iter_trusted_volumes(slice_csv=None):
    """Grouped by volume so each volume is read once: yields
    (kidney_id, img_volume_uint8, gt_volume_bool, spacing_yx, [(z, case_id)])."""
    import csv
    import SimpleITK as sitk

    path = slice_csv or os.path.join(ROOT, "outputs/trusted_us_slices.csv")
    by_kid = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_kid.setdefault(r["kidney_id"], []).append(int(r["slice_idx"]))
    for kid in sorted(by_kid):
        img = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(M.TRUSTED, "US_images", f"{kid}_imgUS.nii.gz")))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(M.TRUSTED, "US_masks", "GT_estimated_masksUS",
                         f"{kid}_maskUS.nii.gz"))) > 0.5
        sp = sitk.ReadImage(os.path.join(M.TRUSTED, "US_images", f"{kid}_imgUS.nii.gz")).GetSpacing()
        slices = [(z, f"{kid}_z{z:03d}") for z in sorted(by_kid[kid])]
        yield kid, img, gt, (float(sp[1]), float(sp[0])), slices


def to_model_input(img):
    """uint8 slice -> window-normalized float [0,1] (SAM/unet family)."""
    from utils_kidney import window_normalize
    return window_normalize(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="unet,medsam,sam_lora,samus")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--datasets", default="okd,trusted")
    ap.add_argument("--prompts", default="p3",
                    help="p3 (E1/E2 headline) or p1_box,p2_point,p3,p4_box (E3)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N cases only")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    models = [m for m in args.models.split(",") if m != "nnunet"]  # nnunet via CLI script
    seeds = [int(s) for s in args.seeds.split(",")]
    prompts = args.prompts.split(",")
    os.makedirs(M.PROMPT_DIR, exist_ok=True)
    import datetime
    with open(os.path.join(ROOT, "outputs/test_access_log.txt"), "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | m3_run_eval | models={models} "
                f"seeds={seeds} datasets={args.datasets} prompts={prompts} limit={args.limit} "
                f"(formal test-set access)\n")

    if "okd" in args.datasets:
        cases = list(M.iter_okd_test())
        if args.limit:
            cases = cases[: args.limit]
        prompt_cache = {k: M.load_prompt(k, "okd") for k in
                        ("p1_box", "p2_point", "p4_box") if any(
                            p.startswith(k[:2]) for p in prompts)}
        for model_name in models:
            for seed in seeds:
                prompted = model_name in ("medsam", "sam_lora", "samus")
                model = None
                if model_name == "unet":
                    model = M.load_unet(seed, device)
                elif model_name == "sam_lora":
                    model = M.load_sam_family("sam_lora", seed, device)
                elif model_name == "medsam":
                    model = M.load_sam_family("medsam", seed, device)
                elif model_name == "samus":
                    model = M.load_samus(seed, device)
                elif model_name == "usfm":
                    model = M.load_usfm(seed, device)
                plist = prompts if prompted else ["none"]
                for cid, img, gt, sp, _meta in cases:
                    for pr in plist:
                        if prompted:
                            box, point, tag = get_prompt(cid, pr, "okd", gt, prompt_cache)
                            if box is None and point is None:
                                d, h = 0.0, float("nan")   # v1.2: empty prompt = Dice 0
                            else:
                                pred = infer_prompted(model_name, model, img, tag, box, point, device)
                                d, h = M.score(pred, gt, sp)
                        elif model_name == "usfm":
                            pred = M.infer_usfm(model, img, device)
                            d, h = M.score(pred, gt, sp)
                            tag = "none"
                        else:
                            pred = infer_unet_case(model, img, device)
                            d, h = M.score(pred, gt, sp)
                            tag = "none"
                        append_row("e1_internal.csv",
                                   ["model", "seed", "prompt", "case_id", "dice", "hd95_px"],
                                   [model_name, seed, tag, cid, f"{d:.6f}", f"{h:.6f}"])
                del model
                torch.cuda.empty_cache()
                print(f"[okd] {model_name} seed{seed} done ({len(cases)} cases x {len(plist)} prompts)",
                      flush=True)

    if "trusted" in args.datasets:
        prompt_cache = {k: M.load_prompt(k, "trusted") for k in
                        ("p1_box", "p2_point", "p4_box") if any(
                            p.startswith(k[:2]) for p in prompts)}
        limit = args.limit
        for model_name in models:
            prompted = model_name in ("medsam", "sam_lora", "samus")
            for seed in seeds:
                model = None
                if model_name == "unet":
                    model = M.load_unet(seed, device)
                elif model_name == "sam_lora":
                    model = M.load_sam_family("sam_lora", seed, device)
                elif model_name == "medsam":
                    model = M.load_sam_family("medsam", seed, device)
                elif model_name == "samus":
                    model = M.load_samus(seed, device)
                elif model_name == "usfm":
                    model = M.load_usfm(seed, device)
                plist = prompts if prompted else ["none"]
                n_done = 0
                for kid, vol, gt, sp, slices in iter_trusted_volumes():
                    for z, cid in slices:
                        if limit and n_done >= limit:
                            break
                        img = vol[z]
                        gt_s = gt[z]
                        for pr in plist:
                            if prompted:
                                box, point, tag = get_prompt(cid, pr, "trusted", gt_s, prompt_cache)
                                if box is None and point is None:
                                    d, h = 0.0, float("nan")
                                else:
                                    pred = infer_prompted(model_name, model, img, tag, box, point, device)
                                    d, h = M.score(pred, gt_s, sp)
                            elif model_name == "usfm":
                                pred = M.infer_usfm(model, img, device)
                                d, h = M.score(pred, gt_s, sp)
                                tag = "none"
                            else:
                                pred = infer_unet_case(model, img, device)
                                d, h = M.score(pred, gt_s, sp)
                                tag = "none"
                            append_row("e2_external.csv",
                                       ["model", "seed", "prompt", "patient", "kidney_id",
                                        "slice_idx", "case_id", "dice", "hd95_mm"],
                                       [model_name, seed, tag, re.sub(r"[LR]$", "", kid), kid, z, cid,
                                        f"{d:.6f}", f"{h:.6f}"])
                            n_done += 1
                    if limit and n_done >= limit:
                        break
                del model
                torch.cuda.empty_cache()
                print(f"[trusted] {model_name} seed{seed} done ({n_done} slices x {len(plist)} prompts)",
                      flush=True)


if __name__ == "__main__":
    main()
