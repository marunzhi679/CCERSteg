import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from pytorch_msssim import ssim

from config_rgb3 import Args
from model import EnhancedHighQualityStegNetUnified


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

args = Args()
device = args.device

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# =====================================================
# Dataset
# =====================================================
class RGBImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        if len(self.files) == 0:
            self.files = sorted(glob.glob(os.path.join(root, "*.png")))
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


train_transform = transforms.Compose([
    transforms.RandomCrop(args.image_size_train),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.CenterCrop(args.image_size_val),
    transforms.ToTensor()
])

train_root = os.path.join(args.data_path, "RGB", "RGB_train")
val_root = os.path.join(args.data_path, "RGB", "RGB_val")

train_loader = DataLoader(
    RGBImageDataset(train_root, train_transform),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    RGBImageDataset(val_root, val_transform),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)


# =====================================================
# Strict secret construction
# 三张 RGB：要求 cover[i], s1[i], s2[i], s3[i] 四者互不相同
# 这要求 batch_size >= 4
# =====================================================
def build_three_rgb_secrets_from_batch(batch: torch.Tensor):
    """
    batch: (B, C, H, W)
    return: s1, s2, s3
    严格保证对每个位置 i:
        s1[i] != cover[i]
        s2[i] != cover[i]
        s3[i] != cover[i]
        s1[i] != s2[i] != s3[i]
    """
    B = batch.size(0)
    device = batch.device

    if B < 4:
        raise ValueError(
            f"Three-RGB strict mode requires batch_size >= 4, but got {B}."
        )

    indices = torch.arange(B, device=device)
    s1_idx = (indices + 1) % B
    s2_idx = (indices + 2) % B
    s3_idx = (indices + 3) % B

    s1 = batch[s1_idx]
    s2 = batch[s2_idx]
    s3 = batch[s3_idx]
    return s1, s2, s3


# =====================================================
# Utils
# =====================================================
def calc_psnr(x, y):
    mse = F.mse_loss(x, y)
    if mse.item() == 0:
        return torch.tensor(100.0, device=x.device)
    return 10 * torch.log10(1.0 / mse)


def get_cover_lambda(epoch, total_epochs, start_v, end_v):
    t = epoch / max(1, total_epochs - 1)
    if t <= 0.8:
        return start_v + (end_v - start_v) * (t / 0.8)
    return end_v


def get_mask_lambda(epoch, total_epochs, max_lambda):
    warm = int(0.08 * total_epochs)
    ramp_end = int(0.25 * total_epochs)
    if epoch < warm:
        return 0.0
    if epoch >= ramp_end:
        return max_lambda
    ratio = (epoch - warm) / max(1, ramp_end - warm)
    return max_lambda * ratio


def get_aux_lambda(epoch, total_epochs, base_lambda):
    t = epoch / max(1, total_epochs - 1)
    if t < 0.15:
        return 0.5 * base_lambda
    elif t < 0.70:
        return 1.0 * base_lambda
    else:
        return 0.7 * base_lambda


def get_ssim_weights(epoch, total_epochs):
    start = int(0.25 * total_epochs)
    end = int(0.60 * total_epochs)
    if epoch < start:
        return 0.0, 0.0, 0.0, 0.0
    if epoch >= end:
        return 0.06, 0.06, 0.06, 0.05
    ratio = (epoch - start) / max(1, end - start)
    return 0.06 * ratio, 0.06 * ratio, 0.06 * ratio, 0.05 * ratio


def get_secret_lambdas(epoch, total_epochs):
    t = epoch / max(1, total_epochs - 1)
    if t <= 0.4:
        ls1 = args.lambda_s1_start - (args.lambda_s1_start - args.lambda_s1_end) * (t / 0.4)
        ls2 = args.lambda_s2_start - (args.lambda_s2_start - args.lambda_s2_end) * (t / 0.4)
        ls3 = args.lambda_s3_start - (args.lambda_s3_start - args.lambda_s3_end) * (t / 0.4)
    else:
        ls1 = args.lambda_s1_end
        ls2 = args.lambda_s2_end
        ls3 = args.lambda_s3_end
    return ls1, ls2, ls3


# =====================================================
# Model
# =====================================================
encoder = EnhancedHighQualityStegNetUnified(args, mode="encoder").to(device)
decoder = EnhancedHighQualityStegNetUnified(args, mode="decoder").to(device)

optimizer = optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=args.cosine_T0,
    T_mult=args.cosine_T_mult,
    eta_min=args.min_lr
)

l1_loss = nn.L1Loss()
best_main_psnr = 0.0
best_avg_psnr = 0.0
start_epoch = 0

# Save the most recent checkpoint separately from the best validation checkpoints.
# Old versions used best_main.pth as the latest checkpoint; support that
# location as a fallback when resuming existing runs.
latest_path = os.path.join(args.save_path, "models", "latest.pth")
legacy_latest_path = os.path.join(args.save_path, "models", "best_main.pth")
resume_path = (
    latest_path if os.path.isfile(latest_path)
    else legacy_latest_path if os.path.isfile(legacy_latest_path)
    else None
)
if resume_path is not None:
    print(f"Resuming checkpoint from: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_main_psnr = ckpt.get("best_main_psnr", 0.0)
    best_avg_psnr = ckpt.get("best_avg_psnr", 0.0)
    print(f"Resume from epoch {start_epoch}")


@torch.no_grad()
def validate(epoch):
    encoder.eval()
    decoder.eval()

    total_cover_psnr = 0.0
    total_s1_psnr = 0.0
    total_s2_psnr = 0.0
    total_s3_psnr = 0.0
    total_cover_ssim = 0.0
    total_s1_ssim = 0.0
    total_s2_ssim = 0.0
    total_s3_ssim = 0.0
    total_batches = 0

    for cover in val_loader:
        cover = cover.to(device)
        s1, s2, s3 = build_three_rgb_secrets_from_batch(cover)

        secrets = [s1, s2, s3]
        valid_mask = torch.ones(cover.size(0), args.num_secrets, device=device)
        secret_type_ids = torch.tensor(args.secret_type_ids, device=device)

        stego, _, _, _ = encoder(
            cover,
            secrets=secrets,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids,
            return_vis=False
        )
        preds, _ = decoder(
            stego,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids
        )

        pred_s1, pred_s2, pred_s3 = preds[0], preds[1], preds[2]

        total_cover_psnr += calc_psnr(stego, cover).item()
        total_s1_psnr += calc_psnr(pred_s1, s1).item()
        total_s2_psnr += calc_psnr(pred_s2, s2).item()
        total_s3_psnr += calc_psnr(pred_s3, s3).item()

        total_cover_ssim += ssim(stego, cover, data_range=1.0).item()
        total_s1_ssim += ssim(pred_s1, s1, data_range=1.0).item()
        total_s2_ssim += ssim(pred_s2, s2, data_range=1.0).item()
        total_s3_ssim += ssim(pred_s3, s3, data_range=1.0).item()
        total_batches += 1

    avg_cover_psnr = total_cover_psnr / total_batches
    avg_s1_psnr = total_s1_psnr / total_batches
    avg_s2_psnr = total_s2_psnr / total_batches
    avg_s3_psnr = total_s3_psnr / total_batches

    avg_cover_ssim = total_cover_ssim / total_batches
    avg_s1_ssim = total_s1_ssim / total_batches
    avg_s2_ssim = total_s2_ssim / total_batches
    avg_s3_ssim = total_s3_ssim / total_batches

    avg_psnr = (avg_cover_psnr + avg_s1_psnr + avg_s2_psnr + avg_s3_psnr) / 4.0

    print(f"\n[Validation] Epoch {epoch}")
    print(f"Cover      PSNR: {avg_cover_psnr:.2f} | SSIM: {avg_cover_ssim:.4f}")
    print(f"Secret1    PSNR: {avg_s1_psnr:.2f} | SSIM: {avg_s1_ssim:.4f}")
    print(f"Secret2    PSNR: {avg_s2_psnr:.2f} | SSIM: {avg_s2_ssim:.4f}")
    print(f"Secret3    PSNR: {avg_s3_psnr:.2f} | SSIM: {avg_s3_ssim:.4f}")
    print(f"Average    PSNR: {avg_psnr:.2f}")

    return avg_s1_psnr, avg_psnr


for epoch in range(start_epoch, args.epochs):
    encoder.train()
    decoder.train()

    lambda_c = get_cover_lambda(epoch, args.epochs, args.lambda_cover_start, args.lambda_cover_end)
    lambda_mask = get_mask_lambda(epoch, args.epochs, args.lambda_mask)
    lambda_aux = get_aux_lambda(epoch, args.epochs, args.lambda_aux)
    lambda_s1, lambda_s2, lambda_s3 = get_secret_lambdas(epoch, args.epochs)
    w_ssim_s1, w_ssim_s2, w_ssim_s3, w_ssim_c = get_ssim_weights(epoch, args.epochs)

    pbar = tqdm(train_loader, total=len(train_loader))

    for cover in pbar:
        cover = cover.to(device)
        s1, s2, s3 = build_three_rgb_secrets_from_batch(cover)

        secrets = [s1, s2, s3]
        valid_mask = torch.ones(cover.size(0), args.num_secrets, device=device)
        secret_type_ids = torch.tensor(args.secret_type_ids, device=device)

        optimizer.zero_grad()

        stego, residual, loss_aux_enc, mask = encoder(
            cover,
            secrets=secrets,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids,
            return_vis=False
        )
        preds, loss_aux_dec = decoder(
            stego,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids
        )

        pred_s1, pred_s2, pred_s3 = preds[0], preds[1], preds[2]

        loss_cover = l1_loss(stego, cover)
        loss_s1 = l1_loss(pred_s1, s1)
        loss_s2 = l1_loss(pred_s2, s2)
        loss_s3 = l1_loss(pred_s3, s3)

        ssim_s1 = 1.0 - ssim(pred_s1, s1, data_range=1.0)
        ssim_s2 = 1.0 - ssim(pred_s2, s2, data_range=1.0)
        ssim_s3 = 1.0 - ssim(pred_s3, s3, data_range=1.0)
        ssim_cover = 1.0 - ssim(stego, cover, data_range=1.0)

        loss_mask = mask.mean()
        loss_aux = loss_aux_enc + loss_aux_dec

        total_loss = (
            lambda_s1 * loss_s1 +
            lambda_s2 * loss_s2 +
            lambda_s3 * loss_s3 +
            lambda_c * loss_cover +
            w_ssim_s1 * ssim_s1 +
            w_ssim_s2 * ssim_s2 +
            w_ssim_s3 * ssim_s3 +
            w_ssim_c * ssim_cover +
            lambda_aux * loss_aux +
            lambda_mask * loss_mask
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            args.grad_clip
        )
        optimizer.step()

        with torch.no_grad():
            cover_psnr = calc_psnr(stego, cover).item()
            s1_psnr = calc_psnr(pred_s1, s1).item()
            s2_psnr = calc_psnr(pred_s2, s2).item()
            s3_psnr = calc_psnr(pred_s3, s3).item()

        pbar.set_description(
            f"Epoch {epoch} | C {cover_psnr:.2f} | S1 {s1_psnr:.2f} | S2 {s2_psnr:.2f} | S3 {s3_psnr:.2f} | λc {lambda_c:.2f}"
        )

    scheduler.step()

    if epoch % args.val_freq == 0:
        main_psnr, avg_psnr = validate(epoch)

        if main_psnr > best_main_psnr:
            best_main_psnr = main_psnr
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_main_psnr": best_main_psnr,
                "best_avg_psnr": best_avg_psnr
            }, os.path.join(args.save_path, "models", "best_main.pth"))
            print(" Saved BEST Secret1 model")

        if avg_psnr > best_avg_psnr:
            best_avg_psnr = avg_psnr
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_main_psnr": best_main_psnr,
                "best_avg_psnr": best_avg_psnr
            }, os.path.join(args.save_path, "models", "best_avg.pth"))
            print(" Saved BEST AVG model")

    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_main_psnr": best_main_psnr,
        "best_avg_psnr": best_avg_psnr
    }, latest_path)