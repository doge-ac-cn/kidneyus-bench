#!/usr/bin/env python3
"""M0 gate 5: MedSAM zero-shot sanity check on 20 random OKD test cases.

Protocol: window-normalized grayscale, resized to 1024x1024 (MedSAM input),
box prompt derived from the GT capsule mask (largest CC) — the standard MedSAM
zero-shot setup. Confirms the GPU inference pipeline end-to-end and records
the reference Dice.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from segment_anything import sam_model_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kidney import dice_score, load_json, set_seed, window_normalize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
MASK_DIR = os.path.join(ROOT, "external_kidneyUS/labels/reviewed_masks_1/capsule")
CKPT = os.path.join(ROOT, "data/weights/medsam_vit_b.pth")
N = 20

if not torch.cuda.is_available():
    # CPU fallback (e.g. when GPUs are occupied by training): the checkpoint was
    # saved from CUDA, so force CPU deserialization
    import functools
    torch.load = functools.partial(torch.load, map_location="cpu")


@torch.no_grad()
def medsam_infer(sam, img_u8, box_xyxy):
    """MedSAM inference flow (see bowang-lab/MedSAM MedSAM.forward).

    box_xyxy is in NATIVE image coordinates and is scaled to the 1024 grid
    here (x by 1024/w, y by 1024/h — the resize is square, per-axis factors).
    v1.2 code review CRITICAL fix: the box was previously passed unscaled,
    which depressed the recorded sanity Dice.
    """
    device = next(sam.parameters()).device
    h, w = img_u8.shape
    img_1024 = np.array(Image.fromarray(img_u8).resize((1024, 1024), Image.BILINEAR))
    box_1024 = np.array([box_xyxy[0] * 1024 / w, box_xyxy[1] * 1024 / h,
                         box_xyxy[2] * 1024 / w, box_xyxy[3] * 1024 / h], dtype=np.float32)
    x = torch.from_numpy(img_1024).float()[None, None].to(device) / 255.0
    x = x.repeat(1, 3, 1, 1)  # SAM ViT stem expects RGB; OKD is grayscale
    x = (x - 0.5) / 0.5  # MedSAM normalization: pixel_values = (x - mean)/std
    with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        image_embedding = sam.image_encoder(x)
        box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=device)[None, :]
        box_torch = box_torch[:, None, :]  # (B, 1, 4) prompt-encoder convention
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None, boxes=box_torch, masks=None)
        low_res_logits, _ = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    low_res = torch.sigmoid(low_res_logits)[0, 0].cpu().numpy()
    mask = np.array(Image.fromarray((low_res > 0.5).astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    return mask


def main():
    set_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=CKPT)
    sam = sam.to(device)
    sam.eval()

    test = load_json(os.path.join(ROOT, "outputs/splits.json"))["test"]
    rng = np.random.default_rng(42)
    picks = sorted(rng.choice(test, size=min(N, len(test)), replace=False).tolist())

    dices = []
    for fn in picks:
        img = np.array(Image.open(os.path.join(IMG_DIR, fn)).convert("L"))
        gt = np.array(Image.open(os.path.join(MASK_DIR, fn))) > 0
        lab, n = ndi.label(gt)
        if n > 1:
            sizes = ndi.sum(gt, lab, range(1, n + 1))
            gt = lab == (int(np.argmax(sizes)) + 1)
        ys, xs = np.where(gt)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        wn = window_normalize(img)
        img_u8 = (wn * 255).astype(np.uint8)
        pred = medsam_infer(sam, img_u8, box)
        d = dice_score(pred, gt)
        dices.append(d)
        print(f"{fn}: dice={d:.4f}")
    print(f"\nMedSAM zero-shot mean Dice over {len(dices)} OKD test images: {np.mean(dices):.4f}")
    with open(os.path.join(ROOT, "outputs/m0_sanity_medsam.json"), "w") as f:
        import json
        json.dump({"per_image": dict(zip(picks, [round(float(d), 4) for d in dices])),
                   "mean_dice": float(np.mean(dices)), "n": len(dices),
                   "checkpoint": "medsam_vit_b.pth (hf-mirror wanglab/medsam-vit-base)",
                   "prompt": "GT-box zero-shot"}, f, indent=2)


if __name__ == "__main__":
    main()
