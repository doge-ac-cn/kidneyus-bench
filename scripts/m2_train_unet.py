#!/usr/bin/env python3
"""M2: lightweight U-Net (M1 in the plan) trained on OKD.

Plan §5-M1: from-scratch control model, 2D grayscale, Dice+BCE loss, AdamW
lr=1e-3, cosine schedule, 100 epochs. 3 seeds (42/43/44). Preprocessing:
window-normalized grayscale resized to 256x256 (mask nearest).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kidney import dice_score, load_json, set_seed, window_normalize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
MASK_DIR = os.path.join(ROOT, "external_kidneyUS/labels/reviewed_masks_1/capsule")
SPLITS = load_json(os.path.join(ROOT, "outputs/splits.json"))
CKPT_DIR = os.path.join(ROOT, "outputs/checkpoints")
LOG_DIR = os.path.join(ROOT, "outputs/logs")
RES = int(os.environ.get("UNET_RES", "256"))


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(cin, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottom = DoubleConv(base * 8, base * 16)
        self.u4 = DoubleConv(base * 16 + base * 8, base * 8)
        self.u3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.u2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.u1 = DoubleConv(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b = self.bottom(self.pool(d4))
        u4 = self.u4(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), d4], 1))
        u3 = self.u3(torch.cat([F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False), d3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False), d2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False), d1], 1))
        return self.head(u1)


class OkdDS(Dataset):
    def __init__(self, filenames, aug=False):
        self.files = filenames
        self.aug = aug  # T1-4 fixed config: hflip/vflip/rot90/bright/contrast, p=0.5 each

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        img = np.array(Image.open(os.path.join(IMG_DIR, fn)).convert("L"))
        msk = np.array(Image.open(os.path.join(MASK_DIR, fn)))
        if self.aug:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = int(np.random.randint(0, 4))
            if k:
                img, msk = np.rot90(img, k), np.rot90(msk, k)
            if np.random.rand() < 0.5:
                img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 255)
            if np.random.rand() < 0.5:
                m = img.mean()
                img = np.clip((img - m) * np.random.uniform(0.8, 1.2) + m, 0, 255)
        img = window_normalize(img)
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((RES, RES), Image.BILINEAR))
        msk = np.array(Image.fromarray(msk.astype(np.uint8) * 255).resize((RES, RES), Image.NEAREST))
        x = torch.from_numpy(img).float()[None] / 255.0
        y = torch.from_numpy((msk > 127).astype(np.float32))[None]
        return x, y


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        pred = (torch.sigmoid(logits.float()) > 0.5)[0, 0].cpu().numpy()
        gt = y[0, 0].cpu().numpy()
        # undo resize back to native mask grid is unnecessary for Dice at 256
        dices.append(dice_score(pred, gt))
    return float(np.mean(dices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=-1)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aug", action="store_true",
                    help="T1-4 aug arm: h/v flip + rot90 + brightness/contrast jitter, p=0.5 each")
    args = ap.parse_args()
    if args.bs < 0:
        args.bs = 16 if args.res <= 256 else 8
    global RES
    RES = args.res
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = "cuda"

    ds_train = OkdDS(SPLITS["train"], aug=args.aug)
    ds_val = OkdDS(SPLITS["val"])
    dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=8,
                          pin_memory=True, drop_last=True,
                          worker_init_fn=lambda wid: np.random.seed(
                              1000 * int(os.environ.get("PYTHON_SEED", str(args.seed))) + wid))
    dl_val = DataLoader(ds_val, batch_size=1, num_workers=4)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # T1-3
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(CKPT_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, (f"unet_aug_seed{args.seed}.log" if args.aug else f"unet_seed{args.seed}.log"))
    best_val = -1.0
    with open(log_path, "a") as log:
        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, y in dl_train:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = 0.5 * soft_dice_loss(logits.float(), y) + 0.5 * bce(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            sched.step()
            val_dice = evaluate(model, dl_val, device)
            line = f"ep{ep:03d} loss={np.mean(losses):.4f} val_dice={val_dice:.4f}"
            print(line); log.write(line + "\n"); log.flush()
            if val_dice > best_val:
                best_val = val_dice
                torch.save({"model": model.state_dict(), "seed": args.seed, "epoch": ep,
                            "val_dice": val_dice, "res": RES},
                           os.path.join(CKPT_DIR, ((f"unet_aug_seed{args.seed}.pt" if args.aug else f"unet_seed{args.seed}.pt") if args.res == 256 else f"unet_res{args.res}_lr{args.lr:g}_seed{args.seed}.pt")))
    print(f"BEST val_dice={best_val:.4f}")


if __name__ == "__main__":
    main()
