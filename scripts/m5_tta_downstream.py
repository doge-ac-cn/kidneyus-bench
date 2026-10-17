#!/usr/bin/env python3
"""Amendment 1 section 4.4 downstream scoring for Round 2.

For each R2 replicate: top-1 box per slice at the registered operating point
(conf >= 0.25) from the replicate's all-detection dump -> the three prompted
models (medsam / sam_lora / samus, registered seeds) -> registered scorer on
all 18,322 slices (empty box -> Dice 0). Reports per model x replicate:
strict Dice, conditional-on-detection Dice, coverage. Round-level criterion
values are means across replicates (frozen wording, section 4.3).

Section 4.4(iv) in-domain non-regression is computed by check_okd() separately.

Frozen decision rule (section 4.4, Round 2): adopt iff ALL of
  (i)   external no-box rate < 25%        (per replicate, mean across reps)
  (ii)  conditional Dice >= 0.75          (each of the three models)
  (iii) strict Dice >= 0.55               (each of the three models)
  (iv)  OKD test det-rate drop <= 2pp AND OKD mAP@0.5 drop <= 0.02
Criteria (ii)/(iii) must hold for ALL three prompted models.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from m3_common import iter_trusted, load_sam_family, load_samus, score, \
    infer_sam_family, infer_samus_box, MODEL_RES  # noqa: E402

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
MODELS = ("medsam", "sam_lora", "samus")
REPS = (42, 43, 44)
CONF_OP = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def strict_conditional(dice_by_case, det_by_case):
    strict = float(np.mean([dice_by_case.get(c, 0.0) for c in dice_by_case]))
    dets = [v for c, v in dice_by_case.items() if det_by_case.get(c)]
    cond = float(np.mean(dets)) if dets else float("nan")
    return strict, cond


def score_replicate(rep, model_name, model_obj):
    dump = np.load(ROOT / f"outputs/prompts/yolo_alldet_trusted_tta_r2_s{rep}.npz")
    case_ids = list(dump["case_ids"])
    per = {}
    for i, b, c in zip(dump["row_index"], dump["boxes"], dump["confs"]):
        per.setdefault(int(i), []).append((float(c), b.tolist()))
    res = MODEL_RES[model_name]
    idx_of = {c: i for i, c in enumerate(case_ids)}
    dice_by, det_by = {}, {}
    n = 0
    for cid, img, gt, spacing, _meta in iter_trusted():
        i = idx_of[cid]
        cands = per.get(i, [])
        sel = [(c, b) for c, b in cands if c >= CONF_OP]
        if not sel:
            dice_by[cid], det_by[cid] = 0.0, False
        else:
            c, b = max(sel, key=lambda x: x[0])
            if model_name == "samus":
                pred = infer_samus_box(model_obj, img, DEVICE, box=b)
            else:
                pred = infer_sam_family(model_obj, img, res, DEVICE, box=b)
            d, _h = score(pred.astype(bool), gt, spacing)
            dice_by[cid], det_by[cid] = float(d), True
        n += 1
        if n % 4000 == 0:
            print(f"  r2 s{rep} {model_name}: {n} slices", flush=True)
    strict, cond = strict_conditional(dice_by, det_by)
    cov = float(np.mean([det_by[c] for c in det_by]))
    return {"strict": round(strict, 4), "conditional": round(cond, 4),
            "coverage": round(cov, 4)}


def check_okd():
    """4.4(iv): OKD test detection rate + AP@0.5, registered vs each R2 replicate."""
    sys.path.insert(0, str(Path(__file__).parent))
    from m3_common import iter_okd_test
    from utils_kidney import box_from_mask
    from ultralytics import YOLO

    cases = [(cid, img, box_from_mask(gt)) for cid, img, gt, _s, _m in iter_okd_test()
             if box_from_mask(gt) is not None]
    import m5_yolo_detect_metrics as M
    out = {}
    for tag, w in [("registered", ROOT / "outputs/yolo_runs/d1_seed43/weights/best.pt")] + [
            (f"r2_s{r}", ROOT / f"outputs/p4tta/r2_s{r}/train/weights/best.pt") for r in REPS]:
        model = YOLO(str(w))
        n_eval = n_det = 0
        preds = []  # (conf, image_idx, iou) - per-candidate, first IoU>=0.5 claims
        for k, (cid, img, gb) in enumerate(cases):
            r = model.predict(np.stack([img] * 3, -1), verbose=False, conf=0.01,
                              device=DEVICE)[0]
            n_eval += 1
            if r.boxes is None or len(r.boxes) == 0:
                continue
            confs = r.boxes.conf.cpu().numpy()
            bs = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            has = bool((confs >= CONF_OP).any())
            n_det += has
            ious = M.iou_vs(bs, np.asarray(gb, dtype=np.float32))
            for c, io in zip(confs, ious):
                preds.append((float(c), k, float(io)))
        ap = M.ap_from_preds(preds, n_eval, 0.5)
        out[tag] = {"det_rate@0.25": round(n_det / n_eval, 4), "ap@0.5": round(ap, 4)}
        print(f"OKD {tag}: {out[tag]}", flush=True)
    base = out["registered"]
    # frozen caliber: round-level criteria are per-criterion means across replicates
    det_mean = sum(out[f"r2_s{r}"]["det_rate@0.25"] for r in REPS) / len(REPS)
    ap_mean = sum(out[f"r2_s{r}"]["ap@0.5"] for r in REPS) / len(REPS)
    drop_pp = (base["det_rate@0.25"] - det_mean) * 100
    drop_ap = base["ap@0.5"] - ap_mean
    out["iv_check"] = {"replicate_det_mean": round(det_mean, 4),
                       "replicate_ap_mean": round(ap_mean, 4),
                       "det_rate_drop_pp": round(drop_pp, 2), "ap_drop": round(drop_ap, 4),
                       "pass(drop<=2pp,ap<=0.02)": bool(drop_pp <= 2.0 and drop_ap <= 0.02)}
    return out


def main():
    only_okd = "--only-okd" in sys.argv
    out_path = ROOT / "outputs/summary/tta_r2_downstream.json"
    doc = json.loads(out_path.read_text()) if out_path.exists() else {}
    if not only_okd:
        for model in MODELS:
            if model not in doc:
                doc[model] = {}
            obj = load_samus(42, DEVICE) if model == "samus" else load_sam_family(model, 42, DEVICE)
            for rep in REPS:
                if str(rep) in doc[model]:
                    continue
                if not (ROOT / f"outputs/prompts/yolo_alldet_trusted_tta_r2_s{rep}.npz").exists():
                    print(f"r2 s{rep} dump not ready - skipped (rerun to backfill)", flush=True)
                    continue
                doc[model][str(rep)] = score_replicate(rep, model, obj)
                out_path.write_text(json.dumps(doc, indent=1))
            del obj
            torch.cuda.empty_cache()
        for model in MODELS:
            strict = float(np.mean([doc[model][str(r)]["strict"] for r in REPS]))
            cond = float(np.mean([doc[model][str(r)]["conditional"] for r in REPS]))
            cov = float(np.mean([doc[model][str(r)]["coverage"] for r in REPS]))
            doc[model]["round2_mean"] = {"strict": round(strict, 4),
                                         "conditional": round(cond, 4),
                                         "coverage": round(cov, 4)}
        out_path.write_text(json.dumps(doc, indent=1))
        print(json.dumps({m: doc[m].get("round2_mean") for m in MODELS}, indent=1))
    if "--okd" in sys.argv or only_okd:
        doc["okd_iv"] = check_okd()
        out_path.write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
