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

from config_rgb2 import Args
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
torch.backends.cudnn.benchmark = False


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
# 两张 RGB：cover[i], s1[i], s2[i] 两两不同
# 要求 batch_size >= 3
# =====================================================
def build_two_rgb_secrets_from_batch(batch: torch.Tensor):
    B = batch.size(0)
    device = batch.device

    if B < 3:
        raise ValueError(
            f"Two-RGB strict mode requires batch_size >= 3, but got {B}."
        )

    indices = torch.arange(B, device=device)
    s1_idx = (indices + 1) % B
    s2_idx = (indices + 2) % B

    s1 = batch[s1_idx]
    s2 = batch[s2_idx]
    return s1, s2


# =====================================================
# Utils
# =====================================================
def calc_psnr(x, y):
    mse = F.mse_loss(x, y)
    if mse.item() == 0:
        return torch.tensor(100.0, device=x.device)
    return 10 * torch.log10(1.0 / mse)


def get_cover_lambda(epoch, total_epochs):
    if epoch < 100:
        return args.lambda_cover_start
    elif epoch < 300:
        ratio = (epoch - 100) / 200.0
        return args.lambda_cover_start + ratio * (args.lambda_cover_mid - args.lambda_cover_start)
    else:
        ratio = min((epoch - 300) / max(1, total_epochs - 300), 1.0)
        return args.lambda_cover_mid + ratio * (args.lambda_cover_end - args.lambda_cover_mid)


def get_mask_lambda(epoch, total_epochs, max_lambda):
    # 前期不用 mask 正则，避免“不嵌”
    if epoch < args.mask_warmup_epochs:
        return 0.0
    warm = args.mask_warmup_epochs
    ramp_end = int(0.35 * total_epochs)
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


def get_secret_lambdas(epoch, total_epochs):
    t = epoch / max(1, total_epochs - 1)
    if t <= 0.5:
        ls1 = args.lambda_s1_start - (args.lambda_s1_start - args.lambda_s1_end) * (t / 0.5)
        ls2 = args.lambda_s2_start - (args.lambda_s2_start - args.lambda_s2_end) * (t / 0.5)
    else:
        ls1 = args.lambda_s1_end
        ls2 = args.lambda_s2_end
    return ls1, ls2


def get_ssim_weights(epoch, total_epochs):
    w_s1 = 0.0
    w_s2 = 0.0
    w_c = 0.0

    if epoch >= args.enable_secret_ssim_after:
        ratio = min((epoch - args.enable_secret_ssim_after) / 120.0, 1.0)
        w_s1 = 0.05 * ratio
        w_s2 = 0.05 * ratio

    if epoch >= args.enable_cover_ssim_after:
        ratio = min((epoch - args.enable_cover_ssim_after) / 150.0, 1.0)
        w_c = 0.03 * ratio

    return w_s1, w_s2, w_c


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

best_secret_mean = 0.0
best_avg_psnr = 0.0
start_epoch = 0

latest_path = os.path.join(args.save_path, "models", "latest.pth")
if os.path.exists(latest_path):
    print("🔁 Resuming checkpoint...")
    ckpt = torch.load(latest_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_secret_mean = ckpt.get("best_secret_mean", 0.0)
    best_avg_psnr = ckpt.get("best_avg_psnr", 0.0)
    print(f"Resume from epoch {start_epoch}")


# =====================================================
# Validation
# =====================================================
@torch.no_grad()
def validate(epoch):
    encoder.eval()
    decoder.eval()

    total_cover_psnr = 0.0
    total_s1_psnr = 0.0
    total_s2_psnr = 0.0
    total_cover_ssim = 0.0
    total_s1_ssim = 0.0
    total_s2_ssim = 0.0
    total_batches = 0

    for cover in val_loader:
        cover = cover.to(device)
        s1, s2 = build_two_rgb_secrets_from_batch(cover)

        secrets = [s1, s2]
        valid_mask = torch.ones(cover.size(0), args.num_secrets, device=device)
        secret_type_ids = torch.tensor(args.secret_type_ids, device=device)

        stego, _, _, _ = encoder(
            cover,
            secrets=secrets,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids,
            return_vis=False
        )
        # Capacity/modality metadata condition the decoder router.
        preds, _ = decoder(
            stego,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids
        )

        pred_s1, pred_s2 = preds[0], preds[1]

        total_cover_psnr += calc_psnr(stego, cover).item()
        total_s1_psnr += calc_psnr(pred_s1, s1).item()
        total_s2_psnr += calc_psnr(pred_s2, s2).item()

        total_cover_ssim += ssim(stego, cover, data_range=1.0).item()
        total_s1_ssim += ssim(pred_s1, s1, data_range=1.0).item()
        total_s2_ssim += ssim(pred_s2, s2, data_range=1.0).item()
        total_batches += 1

    avg_cover_psnr = total_cover_psnr / total_batches
    avg_s1_psnr = total_s1_psnr / total_batches
    avg_s2_psnr = total_s2_psnr / total_batches

    avg_cover_ssim = total_cover_ssim / total_batches
    avg_s1_ssim = total_s1_ssim / total_batches
    avg_s2_ssim = total_s2_ssim / total_batches

    avg_psnr = (avg_cover_psnr + avg_s1_psnr + avg_s2_psnr) / 3.0
    secret_mean = (avg_s1_psnr + avg_s2_psnr) / 2.0

    print(f"\n[Validation] Epoch {epoch}")
    print(f"Cover        PSNR: {avg_cover_psnr:.2f} | SSIM: {avg_cover_ssim:.4f}")
    print(f"Secret1      PSNR: {avg_s1_psnr:.2f} | SSIM: {avg_s1_ssim:.4f}")
    print(f"Secret2      PSNR: {avg_s2_psnr:.2f} | SSIM: {avg_s2_ssim:.4f}")
    print(f"SecretMean   PSNR: {secret_mean:.2f}")
    print(f"Average      PSNR: {avg_psnr:.2f}")

    return avg_s1_psnr, avg_s2_psnr, secret_mean, avg_psnr, avg_cover_psnr


# =====================================================
# Train
# =====================================================
for epoch in range(start_epoch, args.epochs):
    encoder.train()
    decoder.train()

    lambda_c = get_cover_lambda(epoch, args.epochs)
    lambda_mask = get_mask_lambda(epoch, args.epochs, args.lambda_mask_max)
    lambda_aux = get_aux_lambda(epoch, args.epochs, args.lambda_aux)
    lambda_s1, lambda_s2 = get_secret_lambdas(epoch, args.epochs)
    w_ssim_s1, w_ssim_s2, w_ssim_c = get_ssim_weights(epoch, args.epochs)

    pbar = tqdm(train_loader, total=len(train_loader))

    for cover in pbar:
        cover = cover.to(device)
        s1, s2 = build_two_rgb_secrets_from_batch(cover)

        secrets = [s1, s2]
        valid_mask = torch.ones(cover.size(0), args.num_secrets, device=device)
        secret_type_ids = torch.tensor(args.secret_type_ids, device=device)

        optimizer.zero_grad()

        # =====================================================
        # 【关键修正】
        # encoder 返回：stego, residual, aux_loss_total, mask
        # 原代码把第三个当成 mask 了，这是错的
        # =====================================================
        stego, residual, loss_aux_enc, mask = encoder(
            cover,
            secrets=secrets,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids,
            return_vis=False
        )

        # Capacity/modality metadata condition the decoder router.
        preds, loss_aux_dec = decoder(
            stego,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids
        )

        pred_s1, pred_s2 = preds[0], preds[1]

        loss_cover = l1_loss(stego, cover)
        loss_s1 = l1_loss(pred_s1, s1)
        loss_s2 = l1_loss(pred_s2, s2)

        ssim_s1 = 1.0 - ssim(pred_s1, s1, data_range=1.0)
        ssim_s2 = 1.0 - ssim(pred_s2, s2, data_range=1.0)
        ssim_cover = 1.0 - ssim(stego, cover, data_range=1.0)

        loss_mask = mask.mean()
        loss_aux = loss_aux_enc + loss_aux_dec

        total_loss = (
            lambda_s1 * loss_s1 +
            lambda_s2 * loss_s2 +
            lambda_c * loss_cover +
            w_ssim_s1 * ssim_s1 +
            w_ssim_s2 * ssim_s2 +
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
            secret_mean_now = 0.5 * (s1_psnr + s2_psnr)

            pbar.set_description(
                f"Epoch {epoch} | "
                f"C {cover_psnr:.2f} | "
                f"S1 {s1_psnr:.2f} | "
                f"S2 {s2_psnr:.2f} | "
                f"Sm {secret_mean_now:.2f} | "
                f"λs {float(lambda_s1):.1f}/{float(lambda_s2):.1f} | "
                f"λc {float(lambda_c):.2f} | "
                f"λaux {float(lambda_aux):.3f} | "
                f"λm {float(lambda_mask):.4f}"
            )

    scheduler.step()

    if epoch % args.val_freq == 0:
        avg_s1_psnr, avg_s2_psnr, secret_mean, avg_psnr, avg_cover_psnr = validate(epoch)

        if secret_mean > best_secret_mean:
            best_secret_mean = secret_mean
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_secret_mean": best_secret_mean,
                "best_avg_psnr": best_avg_psnr
            }, os.path.join(args.save_path, "models", "best_secret_mean.pth"))
            print(" Saved BEST SecretMean model")

        if avg_psnr > best_avg_psnr:
            best_avg_psnr = avg_psnr
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_secret_mean": best_secret_mean,
                "best_avg_psnr": best_avg_psnr
            }, os.path.join(args.save_path, "models", "best_avg.pth"))
            print(" Saved BEST AVG model")

    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_secret_mean": best_secret_mean,
        "best_avg_psnr": best_avg_psnr
    }, latest_path)
