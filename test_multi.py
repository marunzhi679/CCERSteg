
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import cv2
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_calc

from config_multi import Args
from model import EnhancedHighQualityStegNetUnified


# =====================================================
# 配置
# =====================================================
args = Args()
device = args.device
torch.backends.cudnn.benchmark = True

CKPT_PATH = os.path.join(args.save_path, "models", "latest.pth")
OUT_DIR = "./test_results_multi_native"

# 数据路径（保留原来的路径）
# COVER_DIR = "/media/xd132/USER/ZZS/project/StegFormer/datasets/RGB/RGB_test"
# SECRET_DIR = "/media/xd132/USER/ZZS/project/StegFormer/datasets/RGB/RGB_test"
# IR_DIR = "/media/xd132/USER/ZZS/project/StegFormer/datasets/IR/IR_test"

COVER_DIR = "/media/xd132/USER/MRZ/FreMoEStegformer/图像隐写数据集/test_new/MULTI/cover"
SECRET_DIR = "/media/xd132/USER/MRZ/FreMoEStegformer/图像隐写数据集/test_new/MULTI/RGB"
IR_DIR = "/media/xd132/USER/MRZ/FreMoEStegformer/图像隐写数据集/test_new/MULTI/IR"

PATCH = 512
PATCH_BATCH = 4
MAX_TEST = 50
USE_AMP = True

SECRET_OFFSET = 1

for d in [
    "cover",
    "secret_rgb",
    "ir",
    "stego",
    "revealed_secret",
    "revealed_ir",
    "stego_float32"
]:
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


def load_rgb_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def load_ir_image(path, out_channels):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    else:
        if img.shape[2] == 4:
            img = img[:, :, :3]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    img = img.to(device)
    c = img.shape[1]

    if c == out_channels:
        return img
    if out_channels == 1 and c == 3:
        return img.mean(dim=1, keepdim=True)
    if out_channels == 3 and c == 1:
        return img.repeat(1, 3, 1, 1)
    if out_channels == 2 and c == 1:
        return img.repeat(1, 2, 1, 1)
    if out_channels == 2 and c == 3:
        return img[:, :2]
    if out_channels == 3 and c == 2:
        return torch.cat([img, img[:, :1]], dim=1)

    return img[:, :out_channels]


def save_img(tensor, path):
    x = tensor.squeeze(0).detach().cpu().clamp(0, 1)

    if x.shape[0] == 1:
        img = (
            x.squeeze(0).numpy() * 255.0
        ).round().astype(np.uint8)
        cv2.imwrite(path, img)
    else:
        img = x.permute(1, 2, 0).numpy()
        img = (img * 255.0).round().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)


def clean_state_dict(state_dict):
    cleaned = {}

    for k, v in state_dict.items():
        if k.endswith("total_ops") or k.endswith("total_params"):
            continue
        cleaned[k] = v

    return cleaned


def align_triplet_size(cover, secret_rgb, ir):
    H = min(
        cover.shape[2],
        secret_rgb.shape[2],
        ir.shape[2]
    )
    W = min(
        cover.shape[3],
        secret_rgb.shape[3],
        ir.shape[3]
    )

    cover = cover[:, :, :H, :W]
    secret_rgb = secret_rgb[:, :, :H, :W]
    ir = ir[:, :, :H, :W]

    return cover, secret_rgb, ir


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
            raise RuntimeError(
                f"非法分块大小: {n}"
            )

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
    读取本程序保存的完整 Float32 Stego。
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

    a = img1.squeeze(0).detach().cpu()
    b = img2.squeeze(0).detach().cpu()

    if a.shape[0] == 1:
        im1 = (a.squeeze(0).numpy() * 255.0).astype(np.float32)
        im2 = (b.squeeze(0).numpy() * 255.0).astype(np.float32)
    else:
        im1 = (a.permute(1, 2, 0).numpy() * 255.0).astype(np.float32)
        im2 = (b.permute(1, 2, 0).numpy() * 255.0).astype(np.float32)

    mse = np.mean((im1 - im2) ** 2)
    psnr = 100.0 if mse == 0 else 20.0 * np.log10(
        255.0 / np.sqrt(mse)
    )
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(im1 - im2))

    ssim_val = ssim_calc(
        img1,
        img2,
        data_range=1.0,
        size_average=True
    ).item()

    return psnr, ssim_val, mae, rmse


# =====================================================
# 原来的配对逻辑：不改
# =====================================================
def make_secret_pairs(cover_list, secret_list, offset=1):
    """
    避免 cover 和 secret_rgb 取到同一张图。
    若两个目录相同或文件名高度重合，
    则将 secret 循环平移 offset 位。
    """
    if len(secret_list) == 0:
        raise RuntimeError("secret_list is empty.")

    if len(secret_list) == 1 and len(cover_list) == 1:
        return [(cover_list[0], secret_list[0])]

    same_dir = (
        os.path.abspath(COVER_DIR)
        == os.path.abspath(SECRET_DIR)
    )

    cover_bases = [
        os.path.basename(x)
        for x in cover_list
    ]
    secret_bases = [
        os.path.basename(x)
        for x in secret_list
    ]

    overlap = sum([
        1 for c, s in zip(cover_bases, secret_bases)
        if c == s
    ])

    likely_same_order = (
        overlap >= max(
            1,
            min(len(cover_bases), len(secret_bases)) // 2
        )
    )

    if same_dir or likely_same_order:
        shift = offset % len(secret_list)

        if shift == 0:
            shift = 1

        secret_list = (
            secret_list[shift:] + secret_list[:shift]
        )

    pairs = []
    n = min(len(cover_list), len(secret_list))

    for i in range(n):
        pairs.append(
            (cover_list[i], secret_list[i])
        )

    return pairs


# =====================================================
# 按分块尺寸分组
#
# 保留 PATCH_BATCH 批量推理。
# 不同尺寸的边缘块不能直接 torch.cat。
# =====================================================
def tile_batches(H, W):
    size_groups = {}

    for coord in tile_grid(H, W):
        y0, y1, x0, x1 = coord
        size = (y1 - y0, x1 - x0)

        size_groups.setdefault(
            size, []
        ).append(coord)

    for coords in size_groups.values():
        for start in range(
            0,
            len(coords),
            PATCH_BATCH
        ):
            yield coords[start:start + PATCH_BATCH]


# =====================================================
# 发送端
#
# Cover + RGB + IR -> 完整原尺寸 Stego
# =====================================================
@torch.no_grad()
def encode_patch_batched(
    encoder,
    cover,
    secret_rgb,
    ir
):
    B, _, H, W = cover.shape
    assert B == 1, "当前 multi test 默认 batch_size=1"

    out_stego = torch.empty_like(cover)

    for batch_coords in tile_batches(H, W):
        cover_batch = torch.cat([
            cover[:, :, y0:y1, x0:x1].contiguous()
            for y0, y1, x0, x1 in batch_coords
        ], dim=0)

        secret_batch = torch.cat([
            secret_rgb[:, :, y0:y1, x0:x1].contiguous()
            for y0, y1, x0, x1 in batch_coords
        ], dim=0)

        ir_batch = torch.cat([
            ir[:, :, y0:y1, x0:x1].contiguous()
            for y0, y1, x0, x1 in batch_coords
        ], dim=0)

        valid_mask, secret_type_ids = build_test_meta(
            cover_batch.size(0)
        )

        # 保留原来的 AMP 设置
        with torch.cuda.amp.autocast(
            enabled=(USE_AMP and device.type == "cuda")
        ):
            stego_batch, _, _, _ = encoder(
                cover_batch,
                secrets=[secret_batch, ir_batch],
                valid_mask=valid_mask,
                secret_type_ids=secret_type_ids,
                return_vis=False
            )

        # 非重叠写回完整 Stego
        for k, (y0, y1, x0, x1) in enumerate(batch_coords):
            out_stego[:, :, y0:y1, x0:x1] = (
                stego_batch[k:k+1].float()
            )

    return out_stego


# =====================================================
# 接收端
#
# 只根据完整 Stego 恢复 RGB 和 IR
# 不接收 Cover、Secret 或中间特征
# =====================================================
@torch.no_grad()
def decode_patch_batched(decoder, stego):
    B, C, H, W = stego.shape
    assert B == 1 and C == 3

    outputs = [
        stego.new_empty(
            (B, int(ch), H, W)
        )
        for ch in args.secret_channels
    ]

    for batch_coords in tile_batches(H, W):
        stego_batch = torch.cat([
            stego[:, :, y0:y1, x0:x1].contiguous()
            for y0, y1, x0, x1 in batch_coords
        ], dim=0)

        valid_mask, secret_type_ids = build_test_meta(
            stego_batch.size(0)
        )

        with torch.cuda.amp.autocast(
            enabled=(USE_AMP and device.type == "cuda")
        ):
            preds, _ = decoder(
                stego_batch,
                valid_mask=valid_mask,
                secret_type_ids=secret_type_ids
            )

        if len(preds) != args.num_secrets:
            raise RuntimeError(
                "Decoder 返回的秘密图像数量不正确"
            )

        for k, (y0, y1, x0, x1) in enumerate(batch_coords):
            for sec_idx, out in enumerate(outputs):
                out[:, :, y0:y1, x0:x1] = (
                    preds[sec_idx][k:k+1].float()
                )

    return outputs[0], outputs[1]


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

    ckpt = torch.load(
        CKPT_PATH,
        map_location=device
    )

    if "encoder" in ckpt and "decoder" in ckpt:
        encoder.load_state_dict(
            clean_state_dict(ckpt["encoder"]),
            strict=False
        )
        decoder.load_state_dict(
            clean_state_dict(ckpt["decoder"]),
            strict=False
        )
        print(
            "Loaded encoder/decoder from joint checkpoint."
        )
    else:
        raise RuntimeError(
            "当前 multi test 需要 joint checkpoint（encoder + decoder）。"
        )

    encoder.eval()
    decoder.eval()

    cover_list = list_image_files(COVER_DIR)
    secret_list = list_image_files(SECRET_DIR)
    ir_list = list_image_files(IR_DIR)

    if len(cover_list) == 0:
        raise RuntimeError(
            f"No images found in COVER_DIR: {COVER_DIR}"
        )
    if len(secret_list) == 0:
        raise RuntimeError(
            f"No images found in SECRET_DIR: {SECRET_DIR}"
        )
    if len(ir_list) == 0:
        raise RuntimeError(
            f"No images found in IR_DIR: {IR_DIR}"
        )

    rgb_pairs = make_secret_pairs(
        cover_list,
        secret_list,
        offset=SECRET_OFFSET
    )

    num_test = min(
        len(rgb_pairs),
        len(ir_list),
        MAX_TEST
    )

    print(f"Number of test triplets: {num_test}")
    print(f"SECRET_OFFSET = {SECRET_OFFSET}")

    if os.path.abspath(COVER_DIR) == os.path.abspath(SECRET_DIR):
        print(
            "Detected same COVER_DIR and SECRET_DIR, "
            "secret list has been cyclic-shifted."
        )

    cover_results = []
    secret_results = []
    ir_results = []

    ir_channels = int(
        args.secret_channels[1]
    )

    for idx in tqdm(
        range(num_test),
        desc="Testing Multi Fast"
    ):
        cover_path, secret_path = rgb_pairs[idx]
        ir_path = ir_list[idx]

        cover = load_rgb_image(
            cover_path
        )
        secret_rgb = load_rgb_image(
            secret_path
        )
        ir = load_ir_image(
            ir_path,
            out_channels=ir_channels
        )

        if cover is None or secret_rgb is None or ir is None:
            print(f"Skip invalid sample: {idx}")
            continue

        if cover.shape[1] != 3 or secret_rgb.shape[1] != 3:
            print(
                f"Skip sample {idx}: cover/secret_rgb is not RGB."
            )
            continue

        cover_name = os.path.basename(cover_path)
        secret_name = os.path.basename(secret_path)
        ir_name = os.path.basename(ir_path)

        if cover_name == secret_name:
            print(
                f"Warning: sample {idx} still has same basename "
                f"for cover/secret: {cover_name}"
            )

        cover, secret_rgb, ir = align_triplet_size(
            cover,
            secret_rgb,
            ir
        )

        # ================================================
        # 发送端：原尺寸不重叠编码
        # ================================================
        stego = encode_patch_batched(
            encoder,
            cover,
            secret_rgb,
            ir
        )

        # ================================================
        # 保存、重新读取 Float32 Stego
        # ================================================
        float_path = os.path.join(
            OUT_DIR,
            "stego_float32",
            f"{idx:03d}.pt"
        )

        torch.save(
            stego.detach().cpu().float().contiguous(),
            float_path
        )

        stego = load_received_stego(float_path)

        # ================================================
        # 接收端：仅从 Stego 恢复 RGB 和 IR
        # ================================================
        pred_secret, pred_ir = decode_patch_batched(
            decoder,
            stego
        )

        # ================================================
        # 保留原来的指标计算
        # ================================================
        cover_metric = calc_metrics(
            stego, cover
        )
        secret_metric = calc_metrics(
            pred_secret, secret_rgb
        )
        ir_metric = calc_metrics(
            pred_ir, ir
        )

        cover_results.append(cover_metric)
        secret_results.append(secret_metric)
        ir_results.append(ir_metric)

        fname = f"{idx:03d}.png"

        save_img(
            cover,
            os.path.join(OUT_DIR, "cover", fname)
        )
        save_img(
            secret_rgb,
            os.path.join(OUT_DIR, "secret_rgb", fname)
        )
        save_img(
            ir if ir.shape[1] in [1, 3] else ir[:, :1],
            os.path.join(OUT_DIR, "ir", fname)
        )
        save_img(
            stego,
            os.path.join(OUT_DIR, "stego", fname)
        )
        save_img(
            pred_secret,
            os.path.join(OUT_DIR, "revealed_secret", fname)
        )
        save_img(
            pred_ir if pred_ir.shape[1] in [1, 3] else pred_ir[:, :1],
            os.path.join(OUT_DIR, "revealed_ir", fname)
        )

        print(
            f"[{idx:03d}] "
            f"cover={cover_name} | "
            f"secret={secret_name} | "
            f"ir={ir_name} || "
            f"Cover PSNR: {cover_metric[0]:.2f}, "
            f"SSIM: {cover_metric[1]:.4f} | "
            f"Secret PSNR: {secret_metric[0]:.2f}, "
            f"SSIM: {secret_metric[1]:.4f} | "
            f"IR PSNR: {ir_metric[0]:.2f}, "
            f"SSIM: {ir_metric[1]:.4f}"
        )

    if len(cover_results) == 0:
        print("No valid multi test results.")
        return

    avg_cover = np.mean(
        np.array(cover_results), axis=0
    )
    avg_secret = np.mean(
        np.array(secret_results), axis=0
    )
    avg_ir = np.mean(
        np.array(ir_results), axis=0
    )

    print("\nNative-Resolution Multi Fast Test Results (Float32 receiver)")
    print("------------------------------------------------------------")

    print(
        f"PSNR ↑    | "
        f"Cover: {avg_cover[0]:.4f} | "
        f"Secret: {avg_secret[0]:.4f} | "
        f"IR: {avg_ir[0]:.4f}"
    )
    print(
        f"SSIM ↑    | "
        f"Cover: {avg_cover[1]:.4f} | "
        f"Secret: {avg_secret[1]:.4f} | "
        f"IR: {avg_ir[1]:.4f}"
    )
    print(
        f"MAE  ↓    | "
        f"Cover: {avg_cover[2]:.4f} | "
        f"Secret: {avg_secret[2]:.4f} | "
        f"IR: {avg_ir[2]:.4f}"
    )
    print(
        f"RMSE ↓    | "
        f"Cover: {avg_cover[3]:.4f} | "
        f"Secret: {avg_secret[3]:.4f} | "
        f"IR: {avg_ir[3]:.4f}"
    )

    print("============================================================")


if __name__ == "__main__":
    main()
