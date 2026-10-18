#!/usr/bin/env python3
"""M2: foundation-model fine-tuning on OKD (plan §5-M3 MedSAM, §5-M4 SAM+LoRA).

Shared MedSAM-style protocol: 1024x1024 input, GT-box prompt, Dice+BCE.
  --model medsam    : full fine-tune, AdamW lr=1e-5 (plan M3)
  --model sam_lora  : vision-encoder LoRA r=8 (alpha=16, qkv only) + trainable
                      mask decoder; encoder/prompt-encoder otherwise frozen,
                      AdamW lr=1e-4 (plan M4)
3 seeds each (42/43/44); V100-16GB fp16 AMP.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kidney import box_from_mask, dice_score, load_json, set_seed, window_normalize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
MASK_DIR = os.path.join(ROOT, "external_kidneyUS/labels/reviewed_masks_1/capsule")
SPLITS = load_json(os.path.join(ROOT, "outputs/splits.json"))
CKPT_DIR = os.path.join(ROOT, "outputs/checkpoints")
LOG_DIR = os.path.join(ROOT, "outputs/logs")
RES = 1024

CKPTS = {
    "medsam": os.path.join(ROOT, "data/weights/medsam_vit_b.pth"),
    "sam_lora": os.path.join(ROOT, "data/weights/sam_vit_b_01ec64.pth"),
}


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.lora_A.T) @ self.lora_B.T


def apply_lora(sam, r=8, alpha=16):
    for blk in sam.image_encoder.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha)
    return sam


def patch_decoder_for_batch(sam):
    """Vanilla SAM predict_masks assumes image batch == 1 (repeat_interleave on
    tokens.shape[0]); MedSAM ships a patched decoder. We monkeypatch an
    equivalent batched version (identical math, no token-repeat)."""
    import types

    def batched_predict_masks(self, image_embeddings, image_pe,
                              sparse_prompt_embeddings, dense_prompt_embeddings):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(dense_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        b, c, h, w = image_embeddings.shape
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe.expand(b, -1, -1, -1)
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                         for i in range(self.num_mask_tokens)]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    sam.mask_decoder.predict_masks = types.MethodType(batched_predict_masks, sam.mask_decoder)
    return sam


def load_box_item(fn):
    img = np.array(Image.open(os.path.join(IMG_DIR, fn)).convert("L"))
    msk = np.array(Image.open(os.path.join(MASK_DIR, fn))) > 0
    return load_box_item_from_arrays(img, msk, fn)


def load_box_item_from_arrays(img, msk, fn):
    h, w = img.shape
    wn = window_normalize(img)
    img1024 = np.array(Image.fromarray((wn * 255).astype(np.uint8)).resize((RES, RES), Image.BILINEAR))
    m1024 = np.array(Image.fromarray(msk.astype(np.uint8) * 255).resize((RES, RES), Image.NEAREST)) > 127
    box = box_from_mask(msk)
    if box is None:
        raise ValueError(f"empty capsule mask for {fn} (cohort guarantees non-empty)")
    s = np.array([RES / w, RES / h, RES / w, RES / h], dtype=np.float32)
    box1024 = np.asarray(box, dtype=np.float32) * s
    return img1024, m1024.astype(np.float32), box1024


class OkdBoxDS(Dataset):
    def __init__(self, filenames, aug=False):
        self.files = filenames
        self.aug = aug  # T1-4 fixed config, p=0.5 each (image+mask consistently)

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
            if np.random.rand() < 0.5:
                img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 255)
            if np.random.rand() < 0.5:
                m = img.mean()
                img = np.clip((img - m) * np.random.uniform(0.8, 1.2) + m, 0, 255)
        return load_box_item_from_arrays(img, msk, fn)


def collate_simple(batch):
    imgs = torch.stack([torch.from_numpy(b[0]).float()[None] / 255.0 for b in batch])
    msks = torch.stack([torch.from_numpy(b[1])[None] for b in batch])
    boxes = torch.stack([torch.from_numpy(b[2])[None, :] for b in batch])
    return imgs, msks, boxes


def forward_mask_logits(sam, imgs, boxes, device):
    x = imgs.to(device, non_blocking=True).repeat(1, 3, 1, 1)
    x = (x - 0.5) / 0.5
    with torch.autocast("cuda", dtype=torch.float16):
        emb = sam.image_encoder(x)
        box_t = boxes.to(device)  # (B, 1, 4) — prompt-encoder convention, do not add dims
        sparse, dense = sam.prompt_encoder(points=None, boxes=box_t, masks=None)
        logits, _ = sam.mask_decoder(
            image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False)
    return logits.float()


def soft_dice(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2)) + eps
    den = p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()
def evaluate(sam, loader, device):
    sam.eval()
    dices = []
    for imgs, msks, boxes in loader:
        logits = forward_mask_logits(sam, imgs, boxes, device)
        probs = torch.sigmoid(logits[:, 0])
        probs = F.interpolate(probs[:, None], size=(RES, RES), mode="bilinear",
                              align_corners=False)[:, 0]
        pred = (probs > 0.5).cpu().numpy()
        gt = msks.numpy().astype(bool)
        dices.append([dice_score(pred[i], gt[i]) for i in range(len(gt))])
    return float(np.mean(np.concatenate(dices)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["medsam", "sam_lora"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--aug", action="store_true",
                    help="T1-4 aug arm: h/v flip + rot90 + brightness/contrast jitter, p=0.5 each")
    args = ap.parse_args()
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = "cuda"

    lr = 1e-5 if args.model == "medsam" else 1e-4
    sam = sam_model_registry["vit_b"](checkpoint=CKPTS[args.model])
    if args.model == "sam_lora":
        apply_lora(sam, r=8, alpha=16)
    sam = sam.to(device)
    patch_decoder_for_batch(sam)
    if args.model == "sam_lora":
        for name, p in sam.image_encoder.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False
        for p in sam.prompt_encoder.parameters():
            p.requires_grad = False
    train_params = [p for p in sam.parameters() if p.requires_grad]
    print(f"model={args.model} trainable={sum(p.numel() for p in train_params)/1e6:.2f}M "
          f"lr={lr} bs={args.bs}", flush=True)

    dl_train = DataLoader(OkdBoxDS(SPLITS["train"], aug=args.aug), batch_size=args.bs,
                          shuffle=True, num_workers=6, pin_memory=True, drop_last=True,
                          worker_init_fn=lambda wid: np.random.seed(
                              1000 * int(os.environ.get("PYTHON_SEED", str(args.seed))) + wid),
                          collate_fn=collate_simple)
    dl_val = DataLoader(OkdBoxDS(SPLITS["val"]), batch_size=4, num_workers=4,
                        collate_fn=collate_simple)

    opt = torch.optim.AdamW(train_params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # T1-3
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(CKPT_DIR, exist_ok=True)
    aug_tag = "aug_" if args.aug else ""
    ckpt_path = os.path.join(CKPT_DIR, f"{args.model}_{aug_tag}seed{args.seed}.pt")
    log_path = os.path.join(LOG_DIR, f"{args.model}_{aug_tag}seed{args.seed}.log")
    best_val = -1.0
    with open(log_path, "a") as log:
        for ep in range(1, args.epochs + 1):
            sam.train()
            losses = []
            for imgs, msks, boxes in dl_train:
                imgs, msks, boxes = imgs.to(device), msks.to(device), boxes.to(device)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = forward_mask_logits(sam, imgs, boxes, device)
                logits_up = F.interpolate(logits, size=(RES, RES), mode="bilinear",
                                          align_corners=False)
                loss = 0.5 * soft_dice(logits_up[:, 0], msks[:, 0]) + 0.5 * bce(logits_up, msks[:, :1])
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            sched.step()
            val_dice = evaluate(sam, dl_val, device)
            line = f"ep{ep:02d} loss={np.mean(losses):.4f} val_dice={val_dice:.4f}"
            print(line, flush=True)
            log.write(line + "\n"); log.flush()
            if val_dice > best_val:
                best_val = val_dice
                torch.save({"state_dict": sam.state_dict(), "model_name": args.model,
                            "seed": args.seed, "epoch": ep, "val_dice": val_dice, "res": RES},
                           ckpt_path)
    print(f"BEST val_dice={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
