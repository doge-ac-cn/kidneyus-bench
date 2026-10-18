#!/usr/bin/env python3
"""Supplementary arm: SAMUS fine-tuned with BOX prompts (removes the
point-only-training asymmetry declared in DATA_QUALITY section 8).

Pipeline (identical recipe to the reported point-trained arm, only the
training prompt changes and the prompt encoder is unfrozen):
  - init: official SAMUS checkpoint (same as the reported arm)
  - trainable: the official light modules (cnn_embed, post_pos_embed,
    Adapter, rel_pos 2/5/8/11, upneck) + mask decoder + PROMPT ENCODER
    (previously frozen; box embeddings must be trainable for box prompts)
  - prompts: GT-derived box per training image (same box source that
    MedSAM/SAM+LoRA training used)
  - loss: Dice + BCE with logits, pos_weight=2 (as reported for the arm)
  - AdamW lr 1e-4, cosine over 50 epochs, bs 16 (unchanged)
  - seed 42 point check

Evaluation uses the registered inference pathway (m3_common.infer_samus_box)
on three prompt sets:
  E1 (OKD test, 99):  p3 oracle box / p4 detection box / p2 point
  E2 (TRUSTED, 18,322): p3 oracle box / p4 detection box / p2 point
Outputs outputs/summary/supp_samus_boxft.json (+ .md written by the caller).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import m5_samus as S  # noqa: E402  (reuses load_samus / dataset / collate)
import torch.nn.functional as F  # noqa: E402


def train_forward_box(model, imgs, boxes):
    """Grad-enabled twin of m5_samus.forward_box (that one is @no_grad)."""
    imge = model.image_encoder(imgs)
    se, de = model.prompt_encoder(points=None, boxes=boxes, masks=None)
    low, _ = model.mask_decoder(image_embeddings=imge,
                                image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de,
                                multimask_output=False)
    return F.interpolate(low, (S.RES, S.RES), mode="bilinear", align_corners=False)
from m3_common import iter_okd_test, iter_trusted, infer_samus_box, oracle_box  # noqa: E402
from utils_kidney import dice_score  # noqa: E402

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
_seed = 42
for _i, _a in enumerate(sys.argv):
    if _a == "--seed" and _i + 1 < len(sys.argv):
        _seed = int(sys.argv[_i + 1])
SEED = _seed
OUT = ROOT / f"outputs/summary/supp_samus_boxft_seed{SEED}.json"
CKPT = ROOT / f"outputs/checkpoints/samus_boxft_seed{SEED}.pt"
EPOCHS, BS, LR = 50, 8, 1e-4


def build_and_train(device):
    S.ROOT = str(ROOT)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = S.load_samus(device)
    # unfreeze: light modules + decoder + prompt encoder (box embeddings)
    for p in model.prompt_encoder.parameters():
        p.requires_grad = True
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    KEEP = ("cnn_embed", "post_pos_embed", "Adapter", "upneck",
            "2.attn.rel_pos", "5.attn.rel_pos", "8.attn.rel_pos", "11.attn.rel_pos")
    for n, p in model.image_encoder.named_parameters():
        p.requires_grad = any(k in n for k in KEEP)
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"SAMUS box-FT trainable={sum(p.numel() for p in train_params)/1e6:.2f}M", flush=True)

    dl_train = DataLoader(S.OkdSamusDS(S.SPLITS["train"]), batch_size=BS, shuffle=True,
                          num_workers=6, pin_memory=True, collate_fn=S.collate_samus)
    dl_val = DataLoader(S.OkdSamusDS(S.SPLITS["val"]), batch_size=16,
                        collate_fn=S.collate_samus)
    opt = torch.optim.AdamW(train_params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    best = -1.0
    log = open(ROOT / f"outputs/logs/samus_boxft_seed{SEED}.log", "a")
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for imgs, msks, boxes, _pts in dl_train:
            imgs, msks, boxes = imgs.to(device), msks.to(device), boxes.to(device)
            opt.zero_grad()
            out = train_forward_box(model, imgs, boxes)
            p = torch.sigmoid(out[:, 0])
            num = 2 * (p * msks[:, 0]).sum(dim=(1, 2)) + 1.0
            den = p.sum(dim=(1, 2)) + msks[:, 0].sum(dim=(1, 2)) + 1.0
            loss = 0.5 * (1 - (num / den).mean()) + 0.5 * bce(out, msks[:, :1])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        model.eval()
        with torch.no_grad():
            val = S.evaluate(model, dl_val, device)["box"]
        line = f"ep{ep:02d} loss={np.mean(losses):.4f} val_box={val:.4f}"
        print(line, flush=True)
        log.write(line + "\n")
        if val > best:
            best = val
            torch.save({"state_dict": model.state_dict(), "model_name": "samus",
                        "recipe": "box-ft", "seed": SEED, "val_box": val}, str(CKPT))
        log.flush()
    log.write(f"BEST val_box={best:.4f}\n")
    log.close()
    return best


def eval_e1(model, device):
    p3, p4, pt = [], [], []
    b4 = np.load(ROOT / "outputs/prompts/p4_box_okd.npz", allow_pickle=True)
    b2 = np.load(ROOT / "outputs/prompts/p2_point_okd.npz", allow_pickle=True)
    for fn, img, gt, _sp, _m in iter_okd_test():
        box_or = oracle_box(gt)
        p3.append(dice_score(infer_samus_box(model, img, device, box=box_or), gt))
        bb = b4[fn] if fn in b4.files else None
        if bb is not None and not np.isnan(np.asarray(bb, dtype=float)).any():
            p4.append(dice_score(infer_samus_box(model, img, device, box=np.asarray(bb, dtype=float)), gt))
        else:
            p4.append(0.0)
        pp = b2[fn] if fn in b2.files else None
        if pp is not None and not np.isnan(np.asarray(pp, dtype=float)).any():
            pt.append(dice_score(infer_samus_box(model, img, device, point=np.asarray(pp, dtype=float)), gt))
        else:
            pt.append(0.0)
    return {"p3": round(float(np.mean(p3)), 4), "p4": round(float(np.mean(p4)), 4),
            "p2": round(float(np.mean(pt)), 4), "n": len(p3)}


def eval_e2(model, device):
    z3 = np.load(ROOT / "outputs/prompts/p3_oracle_trusted.npz")
    g3 = {str(c): (None if np.isnan(np.asarray(b, dtype=float)).any()
                   else np.asarray(b, dtype=float).tolist())
          for c, b in zip(z3["case_ids"], z3["boxes"])}
    b4 = np.load(ROOT / "outputs/prompts/p4_box_trusted.npz", allow_pickle=True)
    g4 = {}
    for k in b4.files:
        arr = np.asarray(b4[k], dtype=float)
        g4[k] = None if (arr.size != 4 or np.isnan(arr).any()) else arr.tolist()
    b2 = np.load(ROOT / "outputs/prompts/p2_point_trusted.npz", allow_pickle=True)
    g2 = {}
    for k in b2.files:
        arr = np.asarray(b2[k], dtype=float)
        g2[k] = None if (arr.size != 2 or np.isnan(arr).any()) else arr.tolist()
    res = {p: [] for p in ("p3", "p4", "p2")}
    for i, (cid, img, gt, _sp, _m) in enumerate(iter_trusted()):
        bb = g3.get(cid)
        res["p3"].append(dice_score(infer_samus_box(model, img, device, box=bb), gt) if bb else 0.0)
        bb = g4.get(cid)
        res["p4"].append(dice_score(infer_samus_box(model, img, device, box=bb), gt) if bb else 0.0)
        pp = g2.get(cid)
        res["p2"].append(dice_score(infer_samus_box(model, img, device, point=pp), gt) if pp else 0.0)
        if (i + 1) % 4000 == 0:
            print(f"  E2 {i+1}/18322", flush=True)
    return {k: round(float(np.mean(v)), 4) for k, v in res.items()} | {"n": len(res["p3"])}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    best = build_and_train(device)
    t_train = time.time() - t0
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    model = S.load_samus(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    e1 = eval_e1(model, device)
    print("E1:", e1, flush=True)
    e2 = eval_e2(model, device)
    print("E2:", e2, flush=True)
    refs = {42: {"e1": {"p3": 0.8938, "p4": 0.8835, "p2": 0.8794},
                 "e2": {"p3": 0.7081, "p4": 0.4020, "p2": 0.4777}},
            43: {"e1": {"p3": 0.9126, "p4": 0.9002, "p2": 0.8771},
                 "e2": {"p3": 0.7902, "p4": 0.4258, "p2": 0.4713}}}
    ref = refs.get(SEED, refs[42])
    res = {"arm": "SAMUS box-prompt fine-tune (seed 42, point-check)",
           "recipe": {"epochs": EPOCHS, "lr": LR, "bs": BS,
                      "loss": "Dice + BCEWithLogits(pos_weight=2)",
                      "trainable": "official light modules + mask decoder + prompt encoder",
                      "prompt": "GT-derived box (same source as MedSAM training)"},
           "internal_val_best_box": round(best, 4),
           "e1": e1, "e2": e2,
           "reference_point_trained_arm_seed42": ref,
           "delta_vs_point_trained": {
               "e1_p3": round(e1["p3"] - ref["e1"]["p3"], 4),
               "e1_p4": round(e1["p4"] - ref["e1"]["p4"], 4),
               "e2_p3": round(e2["p3"] - ref["e2"]["p3"], 4),
               "e2_p4": round(e2["p4"] - ref["e2"]["p4"], 4),
               "e2_p2": round(e2["p2"] - ref["e2"]["p2"], 4)},
           "gpu_train_seconds": round(t_train, 1),
           "note": "supplementary arm; not part of the registered main table. "
                   "Evaluation via the registered inference pathway "
                   "(m3_common.infer_samus_box); empty prompt -> Dice 0."}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
