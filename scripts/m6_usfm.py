#!/usr/bin/env python3
"""M6: USFM fine-tune with the OFFICIAL recipe (plan §14 v1.4 A1, corrected).

Official recipe (openmedlab/USFM configs + source):
  - optimizer: AdamW, base_lr 3e-4, weight_decay 0.05, layer-wise lr decay 0.65
  - warmup 20 epochs from 5e-5, then cosine decay to 0
  - 400 epochs, batch_size 64 (we use smaller for V100-16GB)
  - data aug: RandomRotate90(p=0.5)
  - precision: fp16

Protocol deviation from pre-registered A1 (lr=1e-5/50ep/Dice+BCE):
  replaced with official recipe after discovering the pre-registered
  hyperparameters were set without reference to the official fine-tune config.
  Documented as Post-registration Deviation.
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "external_USFM"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if "mmcv._ext" not in sys.modules:
    import types
    _fake_ext = types.ModuleType("mmcv._ext")
    def _missing_op(name):
        if name.startswith("__"):
            raise AttributeError(name)
        def _raise(*a, **k):
            raise RuntimeError(f"compiled op '{name}' unavailable in mmcv-lite")
        return _raise
    _fake_ext.__getattr__ = _missing_op
    sys.modules["mmcv._ext"] = _fake_ext

from utils_kidney import dice_score, load_json, set_seed  # noqa: E402

IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
MASK_DIR = os.path.join(ROOT, "external_kidneyUS/labels/reviewed_masks_1/capsule")
SPLITS = load_json(os.path.join(ROOT, "outputs/splits.json"))
USFM_CKPT = os.path.join(ROOT, "data/weights/USFM_latest.pth")
CKPT_DIR = os.path.join(ROOT, "outputs/checkpoints")
LOG_DIR = os.path.join(ROOT, "outputs/logs")
RES = 512
NUM_LAYERS = 12

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img512_u8):
    x = img512_u8.astype(np.float32) / 255.0
    x = np.stack([x] * 3, axis=0)
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(x)


class OkdUSFMDataset(Dataset):
    def __init__(self, filenames, aug=False):
        self.files = filenames
        self.aug = aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        img = np.array(Image.open(os.path.join(IMG_DIR, fn)).convert("L"))
        msk = np.array(Image.open(os.path.join(MASK_DIR, fn))) > 0
        if self.aug:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = int(np.random.randint(0, 4))
            if k:
                img, msk = np.rot90(img, k), np.rot90(msk, k)
        img512 = np.array(Image.fromarray(img).resize((RES, RES), Image.BILINEAR))
        m512 = np.array(Image.fromarray(msk.astype(np.uint8) * 255).resize((RES, RES), Image.NEAREST)) > 127
        return preprocess_image(img512), torch.from_numpy(m512.astype(np.float32))[None]


def collate_usfm(batch):
    return tuple(torch.stack(x) for x in zip(*batch))


def build_usfm(device):
    from mmseg.registry import MODELS
    from mmengine.registry import DefaultScope
    from usdsgen.modules.backbone.segbackbone import HVITBackbone4Seg  # noqa: F401
    from usdsgen.modules.head.seg.ATMHead import ATMHead  # noqa: F401
    import logging
    from usdsgen.utils.modelutils import load_pretrained

    cfg = dict(
        type="EncoderDecoder",
        backbone=dict(
            type="HVITBackbone4Seg", img_size=RES, patch_size=16,
            embed_dim=768, depth=12, num_heads=12,
            drop_path_rate=0.1, attn_drop_rate=0.0, drop_rate=0.0,
            out_indices=[5, 7, 11], use_abs_pos_emb=False, use_rel_pos_bias=True,
        ),
        decode_head=dict(
            type="ATMHead", img_size=RES, in_channels=768, channels=768,
            num_classes=2, num_layers=3, num_heads=12, use_stages=3, embed_dims=384,
        ),
        train_cfg=dict(), test_cfg=dict(mode="whole"),
    )
    logging.basicConfig(level=logging.WARNING)
    with DefaultScope.overwrite_default_scope("m6usfm_v2"):
        model = MODELS.build(cfg)
    load_pretrained(SimpleNamespace(type="vit", pretrained=USFM_CKPT), model.backbone,
                    logging.getLogger("m6usfm"))
    model = model.to(device)
    return model


# ------------------------------------------------------------------ optimizer
def build_official_param_groups(model, base_lr, weight_decay, layer_decay):
    def get_vit_layer(name):
        if name.startswith("backbone."):
            name = name[len("backbone."):]
        if name.startswith("decode_head"):
            return NUM_LAYERS + 1
        if name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        if name.startswith("patch_embed"):
            return 0
        if name.startswith("rel_pos_bias"):
            return NUM_LAYERS - 1
        if name.startswith("blocks"):
            return int(name.split(".")[1])
        return NUM_LAYERS + 1

    scales = list(layer_decay ** i for i in reversed(range(NUM_LAYERS + 2)))
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            wd = 0.0
        else:
            wd = weight_decay
        layer_id = get_vit_layer(name)
        scale = scales[min(layer_id, len(scales) - 1)]
        gn = f"layer_{layer_id}_{'no_decay' if wd == 0 else 'decay'}"
        if gn not in param_groups:
            param_groups[gn] = {"params": [], "weight_decay": wd,
                                "lr": base_lr * scale, "lr_scale": scale}
        param_groups[gn]["params"].append(param)
    return list(param_groups.values())


class WarmupCosine:
    def __init__(self, optimizer, base_lrs, warmup_epochs, total_epochs, warmup_lr):
        self.optimizer = optimizer
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.warmup_lr = warmup_lr

    def step(self, epoch):
        if epoch < self.warmup:
            lrs = [self.warmup_lr + (bl - self.warmup_lr) * epoch / self.warmup
                   for bl in self.base_lrs]
        else:
            t = (epoch - self.warmup) / (self.total - self.warmup)
            lrs = [bl * 0.5 * (1 + math.cos(math.pi * t)) for bl in self.base_lrs]
        for pg, lr_val in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr_val


# ------------------------------------------------------------------ loss/eval
def infer_logits(model, imgs, device):
    """ATMHead (Mask2Former-style) returns a dict; 'pred' is the semantic
    segmentation probability map (B, num_classes, H, W) — softmax(class) x
    sigmoid(mask), NOT a logit. Returns the foreground probability channel
    (B, 1, H, W)."""
    x = imgs.to(device)
    feats = model.backbone(x)
    out = model.decode_head(feats)
    pred = out["pred"].float()
    if pred.shape[-2:] != (RES, RES):
        pred = F.interpolate(pred, size=(RES, RES), mode="bilinear", align_corners=False)
    return pred[:, 1:2]


def infer_usfm(model, img512_u8, device):
    """Single-image inference for m3_common: uint8 grayscale (512,512) -> bool mask."""
    x = preprocess_image(img512_u8)[None].to(device)  # (1,3,512,512)
    pred_fg = infer_logits(model, x, device)  # (1,1,512,512) probability
    p = pred_fg[0, 0].cpu().numpy()
    return p > 0.5


def dice_bce_loss(pred_fg_prob, mask, eps=1.0):
    num = 2 * (pred_fg_prob * mask).sum(dim=(1, 2)) + eps
    den = pred_fg_prob.sum(dim=(1, 2)) + mask.sum(dim=(1, 2)) + eps
    dice_loss = 1 - (num / den).mean()
    bce_loss = F.binary_cross_entropy(pred_fg_prob.clamp(1e-6, 1 - 1e-6), mask)
    return 0.5 * dice_loss + 0.5 * bce_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices = []
    for imgs, msks in loader:
        imgs = imgs.to(device)
        feats = model.backbone(imgs)
        out = model.decode_head(feats)
        pred = out["pred"]
        if pred.shape[-2:] != (RES, RES):
            pred = F.interpolate(pred, size=(RES, RES), mode="bilinear", align_corners=False)
        prob_fg = pred[:, 1].float()
        p = prob_fg.cpu().numpy()
        gt = msks[:, 0].numpy().astype(bool)
        dices.extend(dice_score(p[i] > 0.5, gt[i]) for i in range(len(gt)))
    return float(np.mean(dices))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train"], default="train")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--layer_decay", type=float, default=0.65)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=20)
    ap.add_argument("--warmup_lr", type=float, default=5e-5)
    ap.add_argument("--tag", default="", help="checkpoint/log suffix, e.g. _regrecipe")
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_usfm(device)

    param_groups = build_official_param_groups(
        model, args.base_lr, args.weight_decay, args.layer_decay)
    trainable = sum(p.numel() for pg in param_groups for p in pg["params"])
    print(f"USFM official recipe: base_lr={args.base_lr} layer_decay={args.layer_decay} "
          f"wd={args.weight_decay} trainable={trainable/1e6:.2f}M", flush=True)

    dl_train = DataLoader(OkdUSFMDataset(SPLITS["train"]), batch_size=args.bs, shuffle=True,
                          num_workers=6, pin_memory=True, collate_fn=collate_usfm)
    dl_val = DataLoader(OkdUSFMDataset(SPLITS["val"]), batch_size=8, collate_fn=collate_usfm)

    import itertools
    opt = torch.optim.AdamW(
        [{"params": pg["params"], "weight_decay": pg["weight_decay"], "lr": pg["lr"]}
         for pg in param_groups], lr=args.base_lr)
    sched = WarmupCosine(opt, None, args.warmup_epochs, args.epochs, args.warmup_lr)
    sched.step(0)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    tag = getattr(args, "tag", "")
    ckpt_path = os.path.join(CKPT_DIR, f"usfm{tag}_seed{args.seed}.pt")
    log_path = os.path.join(LOG_DIR, f"usfm{tag}_seed{args.seed}.log")
    best = -1.0
    with open(log_path, "a") as log:
        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []
            for imgs, msks in dl_train:
                imgs = imgs.to(device)
                msks = msks.to(device)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=torch.cuda.is_available()):
                    feats = model.backbone(imgs)
                    out = model.decode_head(feats)
                pred = out["pred"].float()
                if pred.shape[-2:] != (RES, RES):
                    pred = F.interpolate(pred, size=(RES, RES), mode="bilinear", align_corners=False)
                prob_fg = pred[:, 1].clamp(1e-6, 1 - 1e-6)
                num = 2 * (prob_fg * msks[:, 0]).sum(dim=(1, 2)) + 1.0
                den = prob_fg.sum(dim=(1, 2)) + msks[:, 0].sum(dim=(1, 2)) + 1.0
                dice_loss = 1 - (num / den).mean()
                bce_loss = F.binary_cross_entropy(prob_fg, msks[:, 0])
                loss = 0.5 * dice_loss + 0.5 * bce_loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                all_p = itertools.chain.from_iterable(pg["params"] for pg in opt.param_groups)
                torch.nn.utils.clip_grad_norm_(all_p, 1.0)
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            sched.step(ep - 1)

            val_dice = evaluate(model, dl_val, device)
            line = f"ep{ep:03d} loss={np.mean(losses):.4f} val_dice={val_dice:.4f}"
            print(line, flush=True)
            log.write(line + "\n"); log.flush()
            if val_dice > best:
                best = val_dice
                torch.save({"state_dict": model.state_dict(), "model_name": "usfm",
                            "seed": args.seed, "epoch": ep, "val_dice": val_dice, "res": RES},
                           ckpt_path)
    print(f"BEST val_dice={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
