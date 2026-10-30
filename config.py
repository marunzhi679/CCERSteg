
import os
import torch
from dataclasses import dataclass, field


@dataclass
class Args:
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed: int = 42

    image_size_train: int = 256
    image_size_val: int = 256

    embed_dim: int = 48
    num_experts: int = 6
    window_size: int = 8
    moe_top_k: int = 1

    moe_cue_dim: int = 32
    gradient_stat_dim: int = 24
    cov_stat_dim: int = 8
    cov_window: int = 3
    context_stat_dim: int = 32
    context_dilations: tuple = (1, 2, 4)

    num_secrets: int = 1
    secret_channels: list = field(
        default_factory=lambda: [3]
    )
    secret_type_ids: list = field(
        default_factory=lambda: [0.0]
    )

    epochs: int = 2000
    batch_size: int = 4
    num_workers: int = 4

    lr: float = 2e-4
    weight_decay: float = 1e-5

    val_freq: int = 10
    save_freq: int = 50

    lambda_aux: float = 0.1
    lambda_mask_max: float = 0.01
    lambda_secret_start: float = 30.0
    lambda_secret_end: float = 20.0
    lambda_c_start: float = 1.0
    lambda_c_end: float = 12.0

    cosine_T0: int = 2000
    cosine_T_mult: int = 1
    min_lr: float = 1e-6
    grad_clip: float = 1.0

    data_path: str = field(
        default_factory=lambda: os.environ.get(
            "CCER_DATA_ROOT",
            "/media/xd132/USER/MRZ/FreMoEStegformer/"
            "图像隐写数据集/DIV2K"
        )
    )

    train_image_dir: str = field(
        default_factory=lambda: os.environ.get(
            "CCER_SINGLE_TRAIN_DIR",
            "/media/xd132/USER/MRZ/FreMoEStegformer/"
            "图像隐写数据集/DIV2K/DIV2K_train_HR"
        )
    )

    val_image_dir: str = field(
        default_factory=lambda: os.environ.get(
            "CCER_SINGLE_VAL_DIR",
            "/media/xd132/USER/MRZ/FreMoEStegformer/"
            "图像隐写数据集/DIV2K/DIV2K_valid_HR"
        )
    )

    save_path: str = field(
        default_factory=lambda: os.path.join(
            os.environ.get("CCER_OUTPUT_ROOT", "./outputs"),
            "ckpt_single"
        )
    )

    log_path: str = field(
        default_factory=lambda: os.path.join(
            os.environ.get("CCER_OUTPUT_ROOT", "./outputs"),
            "logs_single"
        )
    )

    def __post_init__(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(
            os.path.join(self.save_path, "models"),
            exist_ok=True
        )
