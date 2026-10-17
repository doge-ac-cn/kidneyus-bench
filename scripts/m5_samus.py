#!/usr/bin/env python3
"""M5: SAMUS (official pretrained ckpt) — zero-shot sanity, prompt-path
evaluation, and decoder-only light fine-tuning on OKD (plan §5-M5).

SAMUS natively consumes POINT prompts (its forward ignores the bbox arg), but
its prompt encoder inherits SAM's box pathway (pretrained). We therefore
evaluate two documented prompt paths:
  box   : direct prompt_encoder(points=None, boxes=...) + mask_decoder
          (protocol-consistent with M3/M4 box prompts)
  point : model.forward native path with the GT-mask centroid point
          (SAMUS-native mode)
Input preprocessing follows SAMUS native convention: resize 256x256 bilinear,
scale /255 (no percentile windowing) — documented deviation from M3/M4.

Usage:
  python m5_samus.py --mode sanity            # 20 test images, both paths (CPU ok)
  python m5_samus.py --mode train --seed 42   # decoder-only light FT (50 ep)
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "external_SAMUS"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_kidney import dice_score, load_json, set_seed  # noqa: E402

IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
MASK_DIR = os.path.join(ROOT, "external_kidneyUS/labels/reviewed_masks_1/capsule")
SPLITS = load_json(os.path.join(ROOT, "outputs/splits.json"))
SAM_CKPT = os.path.join(ROOT, "data/weights/sam_vit_b_01ec64.pth")
SAMUS_CKPT = os.path.join(ROOT, "data/weights/SAMUS.pth")
CKPT_DIR = os.path.join(ROOT, "outputs/checkpoints")
LOG_DIR = os.path.join(ROOT, "outputs/logs")
RES = 256
LOW = 128

ARGS = SimpleNamespace(encoder_input_size=RES, low_image_size=LOW, img_size=RES,
                       ori_size=RES, classes=2, crop=None, sam_ckpt=SAM_CKPT,
                       modelname="SAMUS")


def load_samus(device):
    from models.segment_anything_samus.build_sam_us import samus_model_registry
    model = samus_model_registry["vit_b"](ARGS, checkpoint=SAM_CKPT)
    ckpt = torch.load(SAMUS_CKPT, map_location="cpu", weights_only=False)
    new_state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=True), None
    model.to(device).eval()
    return model


def load_case(fn):
    img = np.array(Image.open(os.path.join(IMG_DIR, fn)).convert("L"))
    msk = np.array(Image.open(os.path.join(MASK_DIR, fn))) > 0
    h, w = img.shape
    img256 = np.array(Image.fromarray(img).resize((RES, RES), Image.BILINEAR))
    m256 = np.array(Image.fromarray(msk.astype(np.uint8) * 255).resize((RES, RES), Image.NEAREST)) > 127
    ys, xs = np.where(msk)
    box256 = np.array([xs.min() * RES / w, ys.min() * RES / h,
                       xs.max() * RES / w, ys.max() * RES / h], dtype=np.float32)
    cy, cx = np.argwhere(msk).mean(axis=0)
    pt = np.array([cx * RES / w, cy * RES / h], dtype=np.float32)
    return img256, m256.astype(np.float32), box256, pt


@torch.no_grad()
def forward_box(model, img_t, box_t):
    imge = model.image_encoder(img_t)
    se, de = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    low, _ = model.mask_decoder(image_embeddings=imge,
                                image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de,
                                multimask_output=False)
    return F.interpolate(low, (RES, RES), mode="bilinear", align_corners=False)


@torch.no_grad()
def forward_point(model, img_t, pt_t, lab_t):
    out = model.forward(img_t, (pt_t, lab_t), None)
    return out["low_res_logits"]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    res = {"box": [], "point": []}
    for imgs, msks, boxes, pts in loader:
        imgs, boxes, pts = (imgs.to(device), boxes.to(device), pts.to(device))
        labels = torch.ones((pts.shape[0], 1), device=device)
        logits_box = forward_box(model, imgs, boxes)
        logits_pt = forward_point(model, imgs, pts[:, None, :], labels)
        for name, lg in (("box", logits_box), ("point", logits_pt)):
            lg = F.interpolate(lg.float(), size=(RES, RES), mode="bilinear", align_corners=False)
            probs = torch.sigmoid(lg[:, 0])
            pred = (probs > 0.5).cpu().numpy()
            gt = msks[:, 0].numpy().astype(bool)
            res[name].extend(dice_score(pred[i], gt[i]) for i in range(len(gt)))
    return {k: float(np.mean(v)) for k, v in res.items()}


class OkdSamusDS(Dataset):
    def __init__(self, filenames):
        self.files = filenames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img256, m256, box256, pt = load_case(self.files[i])
        x = torch.from_numpy(img256).float()[None] / 255.0
        return x, torch.from_numpy(m256)[None], torch.from_numpy(box256), torch.from_numpy(pt)


def collate_samus(batch):
    return tuple(torch.stack(x) for x in zip(*batch))


def run_sanity(device):
    model = load_samus(device)
    test = SPLITS["test"]
    rng = np.random.default_rng(42)
    picks = sorted(rng.choice(test, size=min(20, len(test)), replace=False).tolist())
    loader = DataLoader(OkdSamusDS(picks), batch_size=4, collate_fn=collate_samus)
    res = evaluate(model, loader, device)
    import json
    out = {"mean_dice_box_path": round(res["box"], 4),
           "mean_dice_point_path": round(res["point"], 4),
           "n": len(picks), "checkpoint": "SAMUS.pth (official, sha256 a95de25d...)",
           "note": "box path = SAM-inherited box pathway (protocol-consistent with M3/M4); "
                   "point path = SAMUS-native centroid click"}
    with open(os.path.join(ROOT, "outputs/m0_samus_sanity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def run_train(seed, device, epochs=50, bs=16, lr=1e-4):
    """Decoder-only light fine-tune with native centroid-point prompts (plan M5)."""
    set_seed(seed)
    model = load_samus(device)
    # v2 fine-tune recipe = SAMUS official (Samus.__init__) + decoder:
    #   prompt_encoder frozen; mask_decoder trainable; image_encoder frozen
    #   EXCEPT the US-specific light modules (cnn_embed, post_pos_embed,
    #   Adapter, global-block rel_pos 2/5/8/11, upneck). The earlier
    #   decoder-only variant starved the encoder adapters that carry SAMUS's
    #   ultrasound representations.
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    KEEP = ("cnn_embed", "post_pos_embed", "Adapter", "upneck",
            "2.attn.rel_pos", "5.attn.rel_pos", "8.attn.rel_pos", "11.attn.rel_pos")
    for n, p in model.image_encoder.named_parameters():
        p.requires_grad = any(k in n for k in KEEP)
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"SAMUS light-FT trainable={sum(p.numel() for p in train_params)/1e6:.2f}M", flush=True)

    dl_train = DataLoader(OkdSamusDS(SPLITS["train"]), batch_size=bs, shuffle=True,
                          num_workers=6, pin_memory=True, collate_fn=collate_samus)
    dl_val = DataLoader(OkdSamusDS(SPLITS["val"]), batch_size=16, collate_fn=collate_samus)
    opt = torch.optim.AdamW(train_params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = torch.nn.BCEWithLogitsLoss()

    ckpt_path = os.path.join(CKPT_DIR, f"samus_seed{seed}.pt")
    log_path = os.path.join(LOG_DIR, f"samus_seed{seed}.log")
    best = -1.0
    with open(log_path, "a") as log:
        for ep in range(1, epochs + 1):
            model.train()
            losses = []
            for imgs, msks, boxes, pts in dl_train:
                imgs, msks = imgs.to(device), msks.to(device)
                pts = pts.to(device)
                labels = torch.ones((pts.shape[0], 1), device=device)
                opt.zero_grad()
                out = model.forward(imgs, (pts[:, None, :], labels), None)
                logits = F.interpolate(out["low_res_logits"].float(), (RES, RES),
                                       mode="bilinear", align_corners=False)
                p = torch.sigmoid(logits[:, 0])
                num = 2 * (p * msks[:, 0]).sum(dim=(1, 2)) + 1.0
                den = p.sum(dim=(1, 2)) + msks[:, 0].sum(dim=(1, 2)) + 1.0
                loss = 0.5 * (1 - (num / den).mean()) + 0.5 * bce(logits, msks[:, :1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step()
                losses.append(loss.item())
            sched.step()
            val = evaluate(model, dl_val, device)
            line = f"ep{ep:02d} loss={np.mean(losses):.4f} val_point={val['point']:.4f} val_box={val['box']:.4f}"
            print(line, flush=True)
            log.write(line + "\n"); log.flush()
            if val["point"] > best:  # selection on native path, per §14.6 pre-spec
                best = val["point"]
                torch.save({"state_dict": model.state_dict(), "model_name": "samus",
                            "seed": seed, "epoch": ep, "val_dice_point": val["point"],
                            "val_dice_box": val["box"], "res": RES}, ckpt_path)
    print(f"BEST val_point={best:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "train"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.mode == "train" \
        else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = "cuda"
    set_seed(args.seed if args.mode == "train" else 42)
    if args.mode == "sanity":
        run_sanity(device)
    else:
        run_train(args.seed, device, args.epochs, args.bs)
