#!/usr/bin/env python3
"""M3 shared: model loaders, dataset iterators, prompt handling, metrics.

Protocol invariants (plan §5/§7 + v1.2 review):
  - models are evaluated at their native input resolution, predictions are
    resampled back to the NATIVE GT grid, one shared scorer (dice/HD95)
  - E1/E2 headline = oracle box prompt (P3); E3 adds P1/P2/P4 on both datasets
  - empty prompt (no detection / empty prior) -> Dice = 0, counted in the
    denominator (plan §14 v1.2)
  - HD95 spacing is per-axis in ARRAY order (y, x) for 2D; OKD is reported in
    pixels (spacing 1,1; metadata ambiguous — see RUN_REPORT), TRUSTED in mm
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kidney import box_from_mask, dice_score, hd95_mm, load_json, window_normalize  # noqa: E402

IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
MASK_DIR = os.path.join(ROOT, "external_kidneyUS/labels/reviewed_masks_1/capsule")
TRUSTED = os.path.join(ROOT, "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA")
SPLITS = load_json(os.path.join(ROOT, "outputs/splits.json"))
PROMPT_DIR = os.path.join(ROOT, "outputs/prompts")

MODEL_RES = {"unet": 256, "medsam": 1024, "sam_lora": 1024, "samus": 256}


# ---------------------------------------------------------------- models
def load_unet(seed, device):
    ckpt = torch.load(os.path.join(ROOT, "outputs/checkpoints", f"unet_seed{seed}.pt"),
                      map_location=device, weights_only=False)
    from m2_train_unet import UNet
    model = UNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_sam_family(name, seed, device):
    """name in {medsam, sam_lora}; returns (sam, res)."""
    from segment_anything import sam_model_registry
    from m2_train_foundation import patch_decoder_for_batch
    import m2_train_foundation as mtf

    if device == "cpu":
        # medsam_vit_b.pth carries CUDA storage tags; registry torch.load has no
        # map_location — route through CPU (review: MAJOR)
        import functools
        torch.load = functools.partial(torch.load, map_location="cpu")

    ckpt_path = os.path.join(ROOT, "outputs/checkpoints", f"{name}_seed{seed}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sam = sam_model_registry["vit_b"](checkpoint=(
        os.path.join(ROOT, "data/weights/medsam_vit_b.pth") if name == "medsam"
        else os.path.join(ROOT, "data/weights/sam_vit_b_01ec64.pth")))
    if name == "sam_lora":
        mtf.apply_lora(sam, r=8, alpha=16)
    sam = sam.to(device)
    patch_decoder_for_batch(sam)
    sam.load_state_dict(ckpt["state_dict"])
    sam.eval()
    return sam


def load_usfm(seed, device):
    import m6_usfm
    model = m6_usfm.build_usfm(device)
    ckpt = torch.load(os.path.join(ROOT, "outputs/checkpoints", f"usfm_seed{seed}.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def infer_usfm(model, img, device):
    import m6_usfm as M6
    img512 = np.array(Image.fromarray(img).resize((M6.RES, M6.RES), Image.BILINEAR))
    x = M6.preprocess_image(img512)[None].to(device)  # (1,3,512,512) ImageNet-norm
    pred_fg = M6.infer_logits(model, x, device)  # (1,1,512,512) probability
    p = pred_fg[0, 0].cpu().numpy()
    return np.array(Image.fromarray((p > 0.5).astype(np.uint8) * 255)
                    .resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 127


def load_samus(seed, device):
    import m5_samus
    m5_samus.ROOT = ROOT
    model = m5_samus.load_samus(device)
    ckpt = torch.load(os.path.join(ROOT, "outputs/checkpoints", f"samus_seed{seed}.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------- dataset iterators
def iter_okd_test():
    """Yields (case_id, img_L_uint8, gt_bool, spacing_yx, meta)."""
    for fn in SPLITS["test"]:
        img = np.array(Image.open(os.path.join(IMG_DIR, fn)).convert("L"))
        gt = np.array(Image.open(os.path.join(MASK_DIR, fn))) > 0
        yield fn, img, gt, (1.0, 1.0), {}


def iter_trusted(slice_csv=None):
    """Yields (case_id, img_slice_uint8, gt_slice_bool, spacing_yx, meta) for the
    effective-slice protocol (STAPLE mask area >= 50 px, plan §4).

    case_id = f"{kidney_id}_z{slice_idx:03d}".
    """
    import csv
    import SimpleITK as sitk

    path = slice_csv or os.path.join(ROOT, "outputs/trusted_us_slices.csv")
    by_kid = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_kid.setdefault(r["kidney_id"], []).append(int(r["slice_idx"]))
    for kid in sorted(by_kid):
        img = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(TRUSTED, "US_images", f"{kid}_imgUS.nii.gz")))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(TRUSTED, "US_masks", "GT_estimated_masksUS", f"{kid}_maskUS.nii.gz"))) > 0.5
        sp = sitk.ReadImage(os.path.join(TRUSTED, "US_images", f"{kid}_imgUS.nii.gz")).GetSpacing()
        spacing_yx = (float(sp[1]), float(sp[0]))  # array (y, x) order
        for z in sorted(by_kid[kid]):
            sl = img[z]
            if sl.dtype != np.uint8:  # nifti verified uint8 (0-254); guard for future versions
                sl = ((sl - sl.min()) / max(sl.max() - sl.min(), 1e-6) * 255).astype(np.uint8)
            yield f"{kid}_z{z:03d}", sl, gt[z].copy(), spacing_yx, {"kidney_id": kid, "slice_idx": z}


# ---------------------------------------------------------------- prompt handling
def oracle_box(gt):
    return box_from_mask(gt)


def load_prompt(kind, dataset):
    """kind in {p1_box, p2_point, p4_box}; returns {case_id: array or None}.
    Missing prompts are stored as all-NaN sentinels (np.ma.masked does NOT
    survive a savez/load roundtrip — v1.2 code review)."""
    path = os.path.join(PROMPT_DIR, f"{kind}_{dataset}.npz")
    d = np.load(path, allow_pickle=True)
    out = {}
    for k, v in d.items():
        if np.issubdtype(v.dtype, np.floating) and np.isnan(v).all():
            out[k] = None
        else:
            out[k] = v
    return out


def save_prompt_npz(path, d):
    """None values are stored as all-NaN arrays of the right shape."""
    payload = {}
    for k, v in d.items():
        if v is None:
            payload[k] = np.full(4 if "box" in os.path.basename(path) else 2, np.nan,
                                 dtype=np.float32)
        else:
            payload[k] = np.asarray(v, dtype=np.float32)
    np.savez(path, **payload)


def scale_box(box, res, w, h):
    return np.asarray(box, dtype=np.float32) * np.array([res / w, res / h, res / w, res / h])


def scale_point(pt, res, w, h):
    return np.asarray(pt, dtype=np.float32) * np.array([res / w, res / h])


# ---------------------------------------------------------------- model inference
@torch.no_grad()
def infer_unet(model, img, device):
    x = window_normalize(img)
    x = np.array(Image.fromarray((x * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR))
    x = torch.from_numpy(x).float()[None, None].to(device) / 255.0
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        logits = model(x)
    p = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    pred = np.array(Image.fromarray((p > 0.5).astype(np.uint8) * 255)
                    .resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 127
    return pred


@torch.no_grad()
def infer_sam_family(model, img, res, device, box=None, point=None, bs=1):
    """SAM-family inference for one image. img: uint8 2D. Box in native coords
    (x0,y0,x1,y1) or None; point in native coords (x,y) or None.
    Preprocessing matches training (m2_train_foundation.load_box_item):
    window_normalize -> uint8 -> square resize -> /255 -> (x-0.5)/0.5.
    Returns bool mask at native shape."""
    h, w = img.shape
    wn = window_normalize(img)
    img1024 = np.array(Image.fromarray((wn * 255).astype(np.uint8)).resize((res, res), Image.BILINEAR))
    x = torch.from_numpy(img1024).float()[None, None] / 255.0
    x = x.repeat(1, 3, 1, 1)
    x = (x - 0.5) / 0.5
    x = x.to(device)
    reps = 1  # single-image path; batched multi-image variant can raise this
    x = x.expand(reps, -1, -1, -1)
    if box is not None:
        sb = scale_box(box, res, w, h)
        prompt = torch.as_tensor(sb, dtype=torch.float32, device=device)[None, None, :]
        se, de = model.prompt_encoder(points=None, boxes=prompt, masks=None)
    elif point is not None:
        sp = scale_point(point, res, w, h)
        pts = torch.as_tensor(sp, dtype=torch.float32, device=device)[None, None, :]
        labels = torch.ones((1, 1), dtype=torch.float32, device=device)
        se, de = model.prompt_encoder(points=(pts, labels), boxes=None, masks=None)
    else:
        raise ValueError("needs box or point")
    se = se.expand(reps, -1, -1)
    de = de.expand(reps, -1, -1, -1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        emb = model.image_encoder(x)
        logits, _ = model.mask_decoder(
            image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
            multimask_output=False)
    logits = F.interpolate(logits.float()[:, 0][None], size=(res, res), mode="bilinear",
                           align_corners=False)[0, 0]
    p = torch.sigmoid(logits).cpu().numpy()
    return np.array(Image.fromarray((p > 0.5).astype(np.uint8) * 255)
                    .resize((w, h), Image.NEAREST)) > 127


@torch.no_grad()
def infer_samus_box(model, img, device, box=None, point=None):
    """SAMUS: box -> direct box pathway; point -> native forward. /255 native
    preprocessing, 256 grid. Returns bool mask at native shape."""
    h, w = img.shape
    img256 = np.array(Image.fromarray(img).resize((RES_OF("samus"),) * 2, Image.BILINEAR))
    x = torch.from_numpy(img256).float()[None, None].to(device) / 255.0
    imge = model.image_encoder(x)
    if box is not None:
        sb = scale_box(box, RES_OF("samus"), w, h)
        box_t = torch.as_tensor(sb, dtype=torch.float32, device=device)[None, :]
        se, de = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    else:
        sp = scale_point(point, RES_OF("samus"), w, h)
        pt = torch.as_tensor(sp, dtype=torch.float32, device=device)[None, None, :]
        lab = torch.ones((1, 1), device=device)
        se, de = model.prompt_encoder(points=(pt, lab), boxes=None, masks=None)
    low, _ = model.mask_decoder(image_embeddings=imge,
                                image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                                multimask_output=False)
    p = torch.sigmoid(F.interpolate(low.float(), (RES_OF("samus"), RES_OF("samus")),
                                    mode="bilinear", align_corners=False)[0, 0]).cpu().numpy()
    return np.array(Image.fromarray((p > 0.5).astype(np.uint8) * 255)
                    .resize((w, h), Image.NEAREST)) > 127


def RES_OF(name):
    return MODEL_RES[name]


def score(pred, gt, spacing_yx):
    """Shared scorer: prediction and GT at native grid. pred empty -> Dice 0,
    HD95 NaN (counted by the stats stage)."""
    d = dice_score(pred, gt)
    h = hd95_mm(pred, gt, spacing_yx)
    return d, h


def iou_box(a, b):
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / union) if union > 0 else 0.0
