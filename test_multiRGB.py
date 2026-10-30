
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import importlib.util
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_calc


# =========================
# 路径（直接改这里）
# =========================
CONFIG_PATH = "./config_rgb2.py"   # or ./config_rgb3.py
MODEL_PATH = "/media/xd132/USER/MRZ/github/model.py"

CKPT_PATH = "/media/xd132/USER/MRZ/checkpoint/model_0813/ckpt_rgb2/model1029/latest.pth"

COVER_DIR = "/media/xd132/USER/MRZ/datasets/test/multi/cover"
SECRET_DIR = "/media/xd132/USER/MRZ/datasets/test/multi/secret"

OUT_DIR = "./test_results_multi_RGB2_native"

PATCH = 256
MAX_TEST = 50
SAVE_IMAGES = True


# =========================
# 动态导入
# =========================
def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_module = import_module(CONFIG_PATH, "config")
model_module = import_module(MODEL_PATH, "model")

args = config_module.Args()
device = args.device


# =========================
# 创建输出目录
# =========================
for d in ["cover", "stego", "stego_float32"]:
    os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

for i in range(args.num_secrets):
    os.makedirs(os.path.join(OUT_DIR, f"secret{i+1}"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, f"revealed{i+1}"), exist_ok=True)


# =========================
# 工具函数
# =========================
def list_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
    ])


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def save_img(x, path):
    img = x.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).round().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def align(cover, secrets):
    H = cover.shape[2]
    W = cover.shape[3]

    for s in secrets:
        H = min(H, s.shape[2])
        W = min(W, s.shape[3])

    cover = cover[:, :, :H, :W]
    secrets = [s[:, :, :H, :W] for s in secrets]

    return cover, secrets


# =========================
# 原尺寸不重叠分块
# =========================
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
    读取保存后的 Float32 Stego。
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


def calc_metrics(x, y):
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)

    H = min(x.shape[2], y.shape[2])
    W = min(x.shape[3], y.shape[3])
    x = x[:, :, :H, :W]
    y = y[:, :, :H, :W]

    x_np = (
        x.squeeze(0).detach().cpu()
        .permute(1, 2, 0).numpy() * 255.0
    ).astype(np.float32)

    y_np = (
        y.squeeze(0).detach().cpu()
        .permute(1, 2, 0).numpy() * 255.0
    ).astype(np.float32)

    mse = np.mean((x_np - y_np) ** 2)
    psnr = 100.0 if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))
    ssim = ssim_calc(x, y, data_range=1.0, size_average=True).item()
    mae = np.mean(np.abs(x_np - y_np))
    rmse = np.sqrt(mse)

    return psnr, ssim, mae, rmse


# =========================
# 核心：单 secret_dir -> 多秘密
# 保留原来的选取方式
# =========================
def build_secrets(i, secret_list):
    secrets = []

    for k in range(args.num_secrets):
        idx = (i + k) % len(secret_list)
        secrets.append(secret_list[idx])

    return secrets


# =========================
# 发送端：只生成最终 Stego
# =========================
@torch.no_grad()
def encode_patches(encoder, cover, secrets):
    B, C, H, W = cover.shape
    assert B == 1 and C == 3

    out_s = torch.empty_like(cover)

    valid_mask = torch.ones(
        1, args.num_secrets, device=device
    )
    type_ids = torch.tensor(
        args.secret_type_ids,
        dtype=torch.float32,
        device=device
    )

    for y0, y1, x0, x1 in tile_grid(H, W):
        c = cover[:, :, y0:y1, x0:x1].contiguous()

        s = [
            x[:, :, y0:y1, x0:x1].contiguous()
            for x in secrets
        ]

        stego, _, _, _ = encoder(
            c,
            secrets=s,
            valid_mask=valid_mask,
            secret_type_ids=type_ids,
            return_vis=False,
        )

        if stego.shape != c.shape:
            raise RuntimeError(
                f"Encoder 输出尺寸异常: {stego.shape}"
            )

        out_s[:, :, y0:y1, x0:x1] = stego

    return out_s


# =========================
# 接收端：只使用最终 Stego
# =========================
@torch.no_grad()
def decode_patches(decoder, stego):
    B, C, H, W = stego.shape
    assert B == 1 and C == 3

    valid_mask = torch.ones(
        1, args.num_secrets, device=stego.device
    )
    type_ids = torch.tensor(
        args.secret_type_ids,
        dtype=torch.float32,
        device=stego.device
    )

    # 不使用原始 Secret 创建输出空间
    out_r = [
        stego.new_empty((B, int(ch), H, W))
        for ch in args.secret_channels
    ]

    for y0, y1, x0, x1 in tile_grid(H, W):
        patch = stego[:, :, y0:y1, x0:x1].contiguous()

        preds, _ = decoder(
            patch,
            valid_mask=valid_mask,
            secret_type_ids=type_ids,
        )

        if len(preds) != args.num_secrets:
            raise RuntimeError(
                "Decoder 恢复的秘密图像数量不匹配"
            )

        for k, pred in enumerate(preds):
            expected = (
                B,
                int(args.secret_channels[k]),
                y1 - y0,
                x1 - x0
            )

            if tuple(pred.shape) != expected:
                raise RuntimeError(
                    f"Decoder 输出尺寸异常: {pred.shape}, expected={expected}"
                )

            out_r[k][:, :, y0:y1, x0:x1] = pred

    return out_r


# =========================
# 加载模型
# =========================
encoder = model_module.EnhancedHighQualityStegNetUnified(
    args, "encoder"
).to(device)

decoder = model_module.EnhancedHighQualityStegNetUnified(
    args, "decoder"
).to(device)


# =========================
# Params统计
# =========================
def count_params(model):
    return sum(p.numel() for p in model.parameters())


encoder_params = count_params(encoder)
decoder_params = count_params(decoder)
total_params = encoder_params + decoder_params

print(f"Encoder Params: {encoder_params:,}")
print(f"Decoder Params: {decoder_params:,}")
print(f"Total Params: {total_params:,}")
print(f"Total Params (M): {total_params / 1e6:.2f}M")


ckpt = torch.load(CKPT_PATH, map_location=device)

encoder.load_state_dict(
    ckpt["encoder"], strict=False
)
decoder.load_state_dict(
    ckpt["decoder"], strict=False
)

encoder.eval()
decoder.eval()


# =========================
# 数据
# =========================
cover_list = list_files(COVER_DIR)
secret_list = list_files(SECRET_DIR)

if len(cover_list) == 0:
    raise RuntimeError(
        f"No images found in COVER_DIR: {COVER_DIR}"
    )

if len(secret_list) < args.num_secrets:
    raise RuntimeError(
        f"SECRET_DIR 图像数量不足，至少需要 {args.num_secrets} 张。"
    )

N = min(len(cover_list), MAX_TEST)


# =========================
# 统计量
# =========================
cover_metrics_all = []
secret_metrics_all = [
    [] for _ in range(args.num_secrets)
]


# =========================
# 测试
# =========================
for i in tqdm(range(N), desc="Testing"):
    cover = load_img(cover_list[i])

    secret_paths = build_secrets(i, secret_list)
    secrets = [load_img(p) for p in secret_paths]

    cover, secrets = align(cover, secrets)

    # 发送端：不重叠编码并拼接完整 Stego
    stego = encode_patches(encoder, cover, secrets)

    # 保存完整 Float32 Stego
    float_path = os.path.join(
        OUT_DIR,
        "stego_float32",
        f"{i:03d}.pt"
    )

    torch.save(
        stego.detach().cpu().float().contiguous(),
        float_path
    )

    # 接收端：只从重新读取的 Stego 恢复
    stego = load_received_stego(float_path)
    reveals = decode_patches(decoder, stego)

    c_psnr, c_ssim, c_mae, c_rmse = calc_metrics(
        stego, cover
    )
    cover_metrics_all.append(
        [c_psnr, c_ssim, c_mae, c_rmse]
    )

    msg = (
        f"[{i:03d}] Cover PSNR: {c_psnr:.2f}, "
        f"SSIM: {c_ssim:.4f}, "
        f"MAE: {c_mae:.4f}, RMSE: {c_rmse:.4f}"
    )

    for k in range(args.num_secrets):
        s_psnr, s_ssim, s_mae, s_rmse = calc_metrics(
            reveals[k], secrets[k]
        )

        secret_metrics_all[k].append(
            [s_psnr, s_ssim, s_mae, s_rmse]
        )

        msg += (
            f" | Secret{k+1} PSNR: {s_psnr:.2f}, "
            f"SSIM: {s_ssim:.4f}, "
            f"MAE: {s_mae:.4f}, RMSE: {s_rmse:.4f}"
        )

    print(msg)

    if SAVE_IMAGES:
        fname = f"{i:03d}.png"

        save_img(
            cover,
            os.path.join(OUT_DIR, "cover", fname)
        )
        save_img(
            stego,
            os.path.join(OUT_DIR, "stego", fname)
        )

        for k in range(args.num_secrets):
            save_img(
                secrets[k],
                os.path.join(
                    OUT_DIR, f"secret{k+1}", fname
                )
            )
            save_img(
                reveals[k],
                os.path.join(
                    OUT_DIR, f"revealed{k+1}", fname
                )
            )


# =========================
# 输出平均指标
# =========================
if len(cover_metrics_all) == 0:
    print("No valid test results.")
else:
    cover_avg = np.mean(
        np.array(cover_metrics_all),
        axis=0
    )

    secret_avg_list = []

    for k in range(args.num_secrets):
        secret_avg = np.mean(
            np.array(secret_metrics_all[k]),
            axis=0
        )
        secret_avg_list.append(secret_avg)

    secret_mean_psnr = float(
        np.mean([m[0] for m in secret_avg_list])
    )
    secret_mean_ssim = float(
        np.mean([m[1] for m in secret_avg_list])
    )
    secret_mean_mae = float(
        np.mean([m[2] for m in secret_avg_list])
    )
    secret_mean_rmse = float(
        np.mean([m[3] for m in secret_avg_list])
    )

    overall_psnr = float(
        np.mean(
            [cover_avg[0]] + [m[0] for m in secret_avg_list]
        )
    )
    overall_ssim = float(
        np.mean(
            [cover_avg[1]] + [m[1] for m in secret_avg_list]
        )
    )
    overall_mae = float(
        np.mean(
            [cover_avg[2]] + [m[2] for m in secret_avg_list]
        )
    )
    overall_rmse = float(
        np.mean(
            [cover_avg[3]] + [m[3] for m in secret_avg_list]
        )
    )

    print("\n" + "=" * 72)
    print("Multi-secret Test Results (Float32 independent receiver)")
    print("-" * 72)

    print(
        f"Cover    | PSNR: {cover_avg[0]:.2f} | "
        f"SSIM: {cover_avg[1]:.4f} | "
        f"MAE: {cover_avg[2]:.4f} | "
        f"RMSE: {cover_avg[3]:.4f}"
    )

    for k, avg in enumerate(secret_avg_list, start=1):
        print(
            f"Secret{k}  | PSNR: {avg[0]:.2f} | "
            f"SSIM: {avg[1]:.4f} | "
            f"MAE: {avg[2]:.4f} | RMSE: {avg[3]:.4f}"
        )

    print("-" * 72)

    print(
        f"SecretMean | PSNR: {secret_mean_psnr:.2f} | "
        f"SSIM: {secret_mean_ssim:.4f} | "
        f"MAE: {secret_mean_mae:.4f} | "
        f"RMSE: {secret_mean_rmse:.4f}"
    )

    print(
        f"Overall    | PSNR: {overall_psnr:.2f} | "
        f"SSIM: {overall_ssim:.4f} | "
        f"MAE: {overall_mae:.4f} | "
        f"RMSE: {overall_rmse:.4f}"
    )

    print("=" * 72)
