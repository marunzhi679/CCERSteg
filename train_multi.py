import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim
from thop import profile

from config_multi import Args
from model import EnhancedHighQualityStegNetUnified
from datasets_f import (
    COVER_Dataset, SECRET_Dataset, IR_Dataset,
    transform_train, transform_test
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

args = Args()
device = args.device

best_main_psnr = 0.0
best_avg_psnr = 0.0
start_epoch = 0

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


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


def get_ssim_weight_multi(epoch, total_epochs):
    start = int(0.25 * total_epochs)
    end = int(0.60 * total_epochs)
    if epoch < start:
        return 0.0, 0.0, 0.0
    if epoch >= end:
        return 0.08, 0.06, 0.05
    ratio = (epoch - start) / max(1, end - start)
    return 0.08 * ratio, 0.06 * ratio, 0.05 * ratio


def get_secret_lambdas(epoch, total_epochs):
    t = epoch / max(1, total_epochs - 1)
    if t <= 0.4:
        ls1 = args.lambda_s1_start - (args.lambda_s1_start - args.lambda_s1_end) * (t / 0.4)
        ls2 = args.lambda_s2_start - (args.lambda_s2_start - args.lambda_s2_end) * (t / 0.4)
    else:
        ls1 = args.lambda_s1_end
        ls2 = args.lambda_s2_end
    return ls1, ls2


class EncoderProfileWrapper(nn.Module):
    def __init__(self, encoder, num_secrets, secret_type_ids):
        super().__init__()
        self.encoder = encoder
        self.num_secrets = num_secrets
        self.register_buffer("secret_type_ids_buf", torch.tensor(secret_type_ids, dtype=torch.float32))

    def forward(self, cover, secret_rgb, ir):
        secrets = [secret_rgb, ir]
        valid_mask = torch.ones(cover.size(0), self.num_secrets, device=cover.device)
        secret_type_ids = self.secret_type_ids_buf.to(cover.device)
        stego, _, _, _ = self.encoder(
            cover, secrets=secrets, valid_mask=valid_mask,
            secret_type_ids=secret_type_ids, return_vis=False
        )
        return stego


class DecoderProfileWrapper(nn.Module):
    def __init__(self, decoder, num_secrets, secret_type_ids):
        super().__init__()
        self.decoder = decoder
        self.num_secrets = num_secrets
        self.register_buffer("secret_type_ids_buf", torch.tensor(secret_type_ids, dtype=torch.float32))

    def forward(self, stego):
        valid_mask = torch.ones(stego.size(0), self.num_secrets, device=stego.device)
        secret_type_ids = self.secret_type_ids_buf.to(stego.device)
        preds, _ = self.decoder(stego, valid_mask=valid_mask, secret_type_ids=secret_type_ids)
        return torch.cat(preds, dim=1)


# RGB–IR data loaders use the training configuration
data_root = getattr(args, "data_path", None)  # Or set CCER_DATA_ROOT

cover_train_loader = DataLoader(
    COVER_Dataset(transforms_=transform_train, mode="train", data_root=data_root),
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=args.num_workers,
    drop_last=True
)
secret_train_loader = DataLoader(
    SECRET_Dataset(transforms_=transform_train, mode="train", data_root=data_root),
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=args.num_workers,
    drop_last=True
)
ir_train_loader = DataLoader(
    IR_Dataset(transforms_=transform_train, mode="train", data_root=data_root),
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=args.num_workers,
    drop_last=True
)

cover_val_loader = DataLoader(
    COVER_Dataset(transforms_=transform_test, mode="val", data_root=data_root),
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=args.num_workers,
    drop_last=True
)
secret_val_loader = DataLoader(
    SECRET_Dataset(transforms_=transform_test, mode="val", data_root=data_root),
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=args.num_workers,
    drop_last=True
)
ir_val_loader = DataLoader(
    IR_Dataset(transforms_=transform_test, mode="val", data_root=data_root),
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=args.num_workers,
    drop_last=True
)

encoder = EnhancedHighQualityStegNetUnified(args, mode="encoder").to(device)
decoder = EnhancedHighQualityStegNetUnified(args, mode="decoder").to(device)

with torch.no_grad():
    encoder_profile_model = EncoderProfileWrapper(
        encoder, args.num_secrets, args.secret_type_ids
    ).to(device).eval()
    decoder_profile_model = DecoderProfileWrapper(
        decoder, args.num_secrets, args.secret_type_ids
    ).to(device).eval()

    test_cover = torch.randn(1, 3, args.image_size_train, args.image_size_train).to(device)
    test_secret = torch.randn(1, 3, args.image_size_train, args.image_size_train).to(device)
    test_ir = torch.randn(1, args.secret_channels[1], args.image_size_train, args.image_size_train).to(device)
    test_stego = torch.randn(1, 3, args.image_size_train, args.image_size_train).to(device)

    encoder_mac, encoder_params = profile(
        encoder_profile_model, inputs=(test_cover, test_secret, test_ir), verbose=False
    )
    decoder_mac, decoder_params = profile(
        decoder_profile_model, inputs=(test_stego,), verbose=False
    )

    print("thop result: encoder FLOPs=" + str(encoder_mac * 2) + ", encoder params=" + str(encoder_params))
    print("thop result: decoder FLOPs=" + str(decoder_mac * 2) + ", decoder params=" + str(decoder_params))

optimizer = optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=args.lr, weight_decay=args.weight_decay
)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=args.cosine_T0, T_mult=args.cosine_T_mult, eta_min=args.min_lr
)
l1_loss = nn.L1Loss()

latest_path = os.path.join(args.save_path, "models", "best_avg.pth")
if os.path.exists(latest_path):
    print("🔁 Resuming checkpoint...")
    ckpt = torch.load(latest_path, map_location=device)
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
    total_secret_psnr = 0.0
    total_ir_psnr = 0.0
    total_cover_ssim = 0.0
    total_secret_ssim = 0.0
    total_ir_ssim = 0.0
    total_batches = 0

    for cover, secret_rgb, ir in zip(cover_val_loader, secret_val_loader, ir_val_loader):
        cover = cover.to(device)
        secret_rgb = secret_rgb.to(device)
        ir = ir.to(device)

        secrets = [secret_rgb, ir]
        valid_mask = torch.ones(cover.size(0), args.num_secrets, device=device)
        secret_type_ids = torch.tensor(args.secret_type_ids, device=device)

        stego, _, _, _ = encoder(
            cover, secrets=secrets, valid_mask=valid_mask,
            secret_type_ids=secret_type_ids, return_vis=False
        )
        preds, _ = decoder(stego, valid_mask=valid_mask, secret_type_ids=secret_type_ids)

        pred_secret, pred_ir = preds[0], preds[1]

        total_cover_psnr += calc_psnr(stego, cover).item()
        total_secret_psnr += calc_psnr(pred_secret, secret_rgb).item()
        total_ir_psnr += calc_psnr(pred_ir, ir).item()
        total_cover_ssim += ssim(stego, cover, data_range=1.0).item()
        total_secret_ssim += ssim(pred_secret, secret_rgb, data_range=1.0).item()
        total_ir_ssim += ssim(pred_ir, ir, data_range=1.0).item()
        total_batches += 1

    avg_cover_psnr = total_cover_psnr / total_batches
    avg_secret_psnr = total_secret_psnr / total_batches
    avg_ir_psnr = total_ir_psnr / total_batches
    avg_cover_ssim = total_cover_ssim / total_batches
    avg_secret_ssim = total_secret_ssim / total_batches
    avg_ir_ssim = total_ir_ssim / total_batches
    avg_psnr = (avg_cover_psnr + avg_secret_psnr + avg_ir_psnr) / 3.0

    print(f"\n[Validation] Epoch {epoch}")
    print(f"Cover            PSNR: {avg_cover_psnr:.2f} | SSIM: {avg_cover_ssim:.4f}")
    print(f"Secret (RGB)     PSNR: {avg_secret_psnr:.2f} | SSIM: {avg_secret_ssim:.4f}")
    print(f"IR (Infrared)    PSNR: {avg_ir_psnr:.2f} | SSIM: {avg_ir_ssim:.4f}")
    print(f"Average          PSNR: {avg_psnr:.2f}")

    return avg_secret_psnr, avg_psnr


for epoch in range(start_epoch, args.epochs):
    encoder.train()
    decoder.train()

    lambda_s1, lambda_s2 = get_secret_lambdas(epoch, args.epochs)
    lambda_c = get_cover_lambda(epoch, args.epochs, args.lambda_c_start, args.lambda_c_end)
    w_ssim_s1, w_ssim_s2, w_ssim_c = get_ssim_weight_multi(epoch, args.epochs)
    lambda_aux = get_aux_lambda(epoch, args.epochs, args.lambda_aux)
    lambda_mask = get_mask_lambda(epoch, args.epochs, args.lambda_mask_max)

    total_steps = min(len(cover_train_loader), len(secret_train_loader), len(ir_train_loader))
    pbar = tqdm(zip(cover_train_loader, secret_train_loader, ir_train_loader), total=total_steps)

    for cover, secret_rgb, ir in pbar:
        cover = cover.to(device)
        secret_rgb = secret_rgb.to(device)
        ir = ir.to(device)

        secrets = [secret_rgb, ir]
        valid_mask = torch.ones(cover.size(0), args.num_secrets, device=device)
        secret_type_ids = torch.tensor(args.secret_type_ids, device=device)

        optimizer.zero_grad()

        stego, residual, aux_e, mask = encoder(
            cover, secrets=secrets, valid_mask=valid_mask,
            secret_type_ids=secret_type_ids, return_vis=False
        )
        preds, aux_d = decoder(stego, valid_mask=valid_mask, secret_type_ids=secret_type_ids)
        pred_secret, pred_ir = preds[0], preds[1]

        loss_secret = l1_loss(pred_secret, secret_rgb)
        loss_ir = l1_loss(pred_ir, ir)
        loss_cover = l1_loss(stego, cover)
        loss_aux = aux_e + aux_d

        ssim_secret = 1.0 - ssim(pred_secret, secret_rgb, data_range=1.0)
        ssim_ir = 1.0 - ssim(pred_ir, ir, data_range=1.0)
        ssim_cover = 1.0 - ssim(stego, cover, data_range=1.0)
        loss_mask = mask.mean()

        total_loss = (
            lambda_s1 * loss_secret +
            lambda_s2 * loss_ir +
            lambda_c * loss_cover +
            w_ssim_s1 * ssim_secret +
            w_ssim_s2 * ssim_ir +
            w_ssim_c * ssim_cover +
            lambda_aux * loss_aux +
            lambda_mask * loss_mask
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), args.grad_clip
        )
        optimizer.step()

        with torch.no_grad():
            cover_psnr = calc_psnr(stego, cover).item()
            secret_psnr = calc_psnr(pred_secret, secret_rgb).item()
            ir_psnr = calc_psnr(pred_ir, ir).item()

        pbar.set_description(
            f"Epoch {epoch} | Cover {cover_psnr:.2f} | Secret {secret_psnr:.2f} | IR {ir_psnr:.2f} | λc {lambda_c:.2f} | λm {lambda_mask:.4f}"
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
            print(" Saved BEST Secret (RGB) model")

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
