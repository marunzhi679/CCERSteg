
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import cv2
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_calc

from config import Args
from model import EnhancedHighQualityStegNetUnified


# =====================================================
# 配置
# =====================================================
args = Args()
device = args.device

CKPT_PATH = "/media/xd132/USER/MRZ/Stegformer/checkpoint/ckpt_single0813/latest.pth"
OUT_DIR = "./test_results_single_native"

TEST_DATA_DIR = "/media/xd132/USER/MRZ/datasets/test"
COVER_DIR = os.path.join(TEST_DATA_DIR, "cover_AID")
SECRET_DIR = os.path.join(TEST_DATA_DIR, "secret_AID")

# AID 数据集：
# COVER_DIR = os.path.join(TEST_DATA_DIR, "cover_AID")
# SECRET_DIR = os.path.join(TEST_DATA_DIR, "secret_AID")

PATCH = 256
MAX_TEST = 50

for d in ["cover", "secret", "stego", "revealed", "stego_float32"]:
    os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)


# =====================================================
# 工具函数
# =====================================================
def list_image_files(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]
    return sorted(files)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


def save_img(tensor, path):
    img = tensor.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def align_pair_size(cover, secret):
    """
    保留原来的尺寸对齐方式。
    """
    H = min(cover.shape[2], secret.shape[2])
    W = min(cover.shape[3], secret.shape[3])
    cover = cover[:, :, :H, :W]
    secret = secret[:, :, :H, :W]
    return cover, secret


# =====================================================
# 原尺寸非重叠分块
# =====================================================
TILE_MULTIPLE = 8
MIN_TILE_SIZE = 32


def tile_intervals(length):
    if length < MIN_TILE_SIZE or length % TILE_MULTIPLE:
        raise ValueError(
            f"图像边长 {length} 不符合要求：至少 {MIN_TILE_SIZE} "
            f"且须是 {TILE_MULTIPLE} 的倍数；本协议不自动填充。"
        )

    if PATCH % TILE_MULTIPLE or MIN_TILE_SIZE > PATCH:
        raise ValueError(
            "PATCH 必须是 TILE_MULTIPLE 的倍数，且不小于 MIN_TILE_SIZE"
        )

    sizes = [PATCH] * (length // PATCH)
    rem = length % PATCH

    if rem:
        sizes.append(rem)

    if sizes[-1] < MIN_TILE_SIZE:
        if len(sizes) < 2:
            raise ValueError(
                f"图像边长 {length} 不能按当前规则分块"
            )

        total = sizes[-2] + sizes[-1]
        a = total // (2 * TILE_MULTIPLE) * TILE_MULTIPLE
        sizes[-2:] = [a, total - a]

    out = []
    start = 0

    for n in sizes:
        if n < MIN_TILE_SIZE or n > PATCH or n % TILE_MULTIPLE:
            raise RuntimeError(f"非法分块大小: {n}")

        out.append((start, start + n))
        start += n

    assert start == length
    return out


def tile_grid(H, W):
    for y0, y1 in tile_intervals(H):
        for x0, x1 in tile_intervals(W):
            yield y0, y1, x0, x1


def load_received_stego(path):
    """
    接收端读取当前程序保存的 Float32 Stego。
    """
    try:
        tensor = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        tensor = torch.load(path, map_location="cpu")

    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        raise RuntimeError(
            f"Stego 文件不是 Float32 张量: {path}"
        )

    return tensor.to(device)


def build_test_meta(batch_size):
    valid_mask = torch.ones(
        batch_size,
        args.num_secrets,
        device=device
    )

    secret_type_ids = torch.tensor(
        args.secret_type_ids,
        dtype=torch.float32,
        device=device
    )

    return valid_mask, secret_type_ids


def calc_metrics(img1, img2):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    H = min(img1.shape[2], img2.shape[2])
    W = min(img1.shape[3], img2.shape[3])
    img1 = img1[:, :, :H, :W]
    img2 = img2[:, :, :H, :W]

    im1 = (
        img1.squeeze(0)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).astype(np.float32)

    im2 = (
        img2.squeeze(0)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).astype(np.float32)

    mse = np.mean((im1 - im2) ** 2)
    psnr = 100.0 if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(im1 - im2))

    ssim_val = ssim_calc(
        img1,
        img2,
        data_range=1.0,
        size_average=True
    ).item()

    return psnr, ssim_val, mae, rmse


def clean_state_dict(state_dict):
    cleaned = {}

    for k, v in state_dict.items():
        if k.endswith("total_ops") or k.endswith("total_params"):
            continue
        cleaned[k] = v

    return cleaned


# =====================================================
# 参数量统计
# =====================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# =====================================================
# 发送端：只生成最终 Stego
# =====================================================
@torch.no_grad()
def encode_patches(encoder, cover, secret):
    B, C, H, W = cover.shape
    assert B == 1 and C == 3 and secret.shape == cover.shape

    out_stego = torch.empty_like(cover)

    valid_mask, secret_type_ids = build_test_meta(B)

    for y0, y1, x0, x1 in tile_grid(H, W):
        c_patch = cover[:, :, y0:y1, x0:x1].contiguous()
        s_patch = secret[:, :, y0:y1, x0:x1].contiguous()

        stego_patch, _, _, _ = encoder(
            c_patch,
            secrets=[s_patch],
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids,
            return_vis=False
        )

        if stego_patch.shape != c_patch.shape:
            raise RuntimeError(
                f"Encoder 输出尺寸异常: {stego_patch.shape}"
            )

        out_stego[:, :, y0:y1, x0:x1] = stego_patch

    return out_stego


# =====================================================
# 接收端：只使用最终 Stego
# =====================================================
@torch.no_grad()
def decode_patches(decoder, stego):
    B, C, H, W = stego.shape
    assert B == 1 and C == 3

    valid_mask, secret_type_ids = build_test_meta(B)

    out_reveal = stego.new_empty(
        (B, int(args.secret_channels[0]), H, W)
    )

    for y0, y1, x0, x1 in tile_grid(H, W):
        patch = stego[:, :, y0:y1, x0:x1].contiguous()

        preds, _ = decoder(
            patch,
            valid_mask=valid_mask,
            secret_type_ids=secret_type_ids
        )

        pred = preds[0]

        expected = (
            B,
            int(args.secret_channels[0]),
            y1 - y0,
            x1 - x0
        )

        if tuple(pred.shape) != expected:
            raise RuntimeError(
                f"Decoder 输出尺寸异常: {pred.shape}, expected={expected}"
            )

        out_reveal[:, :, y0:y1, x0:x1] = pred

    return out_reveal


# =====================================================
# 主程序
# =====================================================
def main():
    print(f"Using device: {device}")
    print(f"Loading checkpoint from: {CKPT_PATH}")

    encoder = EnhancedHighQualityStegNetUnified(
        args,
        mode="encoder"
    ).to(device)

    decoder = EnhancedHighQualityStegNetUnified(
        args,
        mode="decoder"
    ).to(device)

    # 计算并输出参数量
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    total_params = encoder_params + decoder_params

    print("\n" + "=" * 64)
    print("Model Parameter Count")
    print("-" * 64)
    print(f"Encoder parameters : {encoder_params:,} ({encoder_params / 1e6:.2f}M)")
    print(f"Decoder parameters : {decoder_params:,} ({decoder_params / 1e6:.2f}M)")
    print(f"Total parameters   : {total_params:,} ({total_params / 1e6:.2f}M)")
    print("=" * 64 + "\n")

    ckpt = torch.load(CKPT_PATH, map_location=device)

    if "encoder" in ckpt and "decoder" in ckpt:
        encoder_state = clean_state_dict(ckpt["encoder"])
        decoder_state = clean_state_dict(ckpt["decoder"])

        msg1 = encoder.load_state_dict(encoder_state, strict=False)
        msg2 = decoder.load_state_dict(decoder_state, strict=False)

        print("Loaded encoder/decoder from joint checkpoint.")
        print("Encoder load msg:", msg1)
        print("Decoder load msg:", msg2)
    else:
        raise RuntimeError(
            "当前 test 需要包含 encoder 和 decoder 的 joint checkpoint。"
        )

    encoder.eval()
    decoder.eval()

    cover_list = list_image_files(COVER_DIR)
    secret_list = list_image_files(SECRET_DIR)

    if len(cover_list) == 0:
        raise RuntimeError(
            f"No images found in COVER_DIR: {COVER_DIR}"
        )
    if len(secret_list) == 0:
        raise RuntimeError(
            f"No images found in SECRET_DIR: {SECRET_DIR}"
        )

    num_test = min(len(cover_list), len(secret_list), MAX_TEST)
    print(f"Number of test pairs: {num_test}")

    stego_results = []
    reveal_results = []

    with torch.no_grad():
        for idx in tqdm(range(num_test), desc="Testing"):
            cover = load_image(cover_list[idx])
            secret = load_image(secret_list[idx])

            if cover is None or secret is None:
                print(f"Skip invalid pair: {idx}")
                continue

            if cover.shape[1] != 3 or secret.shape[1] != 3:
                print(
                    f"Skip pair {idx}: cover/secret is not 3-channel RGB."
                )
                continue

            cover, secret = align_pair_size(cover, secret)

            # 发送端：原尺寸不重叠编码
            stego = encode_patches(encoder, cover, secret)

            # 保存完整 Float32 Stego
            float_path = os.path.join(
                OUT_DIR,
                "stego_float32",
                f"{idx:03d}.pt"
            )

            torch.save(
                stego.detach().cpu().float().contiguous(),
                float_path
            )

            # 接收端：重新读取 Stego 并独立恢复
            received_stego = load_received_stego(float_path)
            reveal = decode_patches(decoder, received_stego)
            stego = received_stego

            stego_metric = calc_metrics(stego, cover)
            reveal_metric = calc_metrics(reveal, secret)

            stego_results.append(stego_metric)
            reveal_results.append(reveal_metric)

            fname = f"{idx:03d}.png"

            save_img(
                cover,
                os.path.join(OUT_DIR, "cover", fname)
            )
            save_img(
                secret,
                os.path.join(OUT_DIR, "secret", fname)
            )
            save_img(
                stego,
                os.path.join(OUT_DIR, "stego", fname)
            )
            save_img(
                reveal,
                os.path.join(OUT_DIR, "revealed", fname)
            )

            print(
                f"[{idx:03d}] "
                f"Stego PSNR: {stego_metric[0]:.2f}, "
                f"SSIM: {stego_metric[1]:.4f} | "
                f"Reveal PSNR: {reveal_metric[0]:.2f}, "
                f"SSIM: {reveal_metric[1]:.4f}"
            )

    if len(stego_results) == 0:
        print("No valid test results.")
        return

    avg_s = np.mean(np.array(stego_results), axis=0)
    avg_r = np.mean(np.array(reveal_results), axis=0)

    print("\n" + "=" * 64)
    print("Native-Resolution Single Test Results (Float32 receiver)")
    print("-" * 64)

    names = ["PSNR ↑", "SSIM ↑", "MAE ↓", "RMSE ↓"]

    for i in range(4):
        print(
            f"{names[i]:<10} | "
            f"Stego: {avg_s[i]:.4f} | "
            f"Reveal: {avg_r[i]:.4f}"
        )

    print("=" * 64)


if __name__ == "__main__":
    main()
