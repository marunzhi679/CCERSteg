import os
import torch
from dataclasses import dataclass, field


@dataclass
class Args:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    image_size_train: int = 256
    image_size_val: int = 256

    embed_dim: int = 48
    num_experts: int = 6
    window_size: int = 8
    moe_top_k: int = 1

    # RGB + IR
    num_secrets: int = 2
    secret_channels: list = field(default_factory=lambda: [3, 3])   # 当前 datasets_f.py 中 IR 仍按 3 通道读入
    secret_type_ids: list = field(default_factory=lambda: [0.0, 1.0])  # RGB, IR

    epochs: int = 1200
    # batch size 由 train 中的 DataLoader 直接读取本配置
    batch_size: int = 2
    num_workers: int = 8

    lr: float = 2e-4
    weight_decay: float = 1e-5

    val_freq: int = 10
    save_freq: int = 50

    lambda_aux: float = 0.1
    lambda_mask_max: float = 0.01
    lambda_s1_start: float = 24.0
    lambda_s1_end: float = 18.0
    lambda_s2_start: float = 18.0
    lambda_s2_end: float = 14.0
    lambda_c_start: float = 1.0
    lambda_c_end: float = 10.0

    cosine_T0: int = 2000
    cosine_T_mult: int = 1
    min_lr: float = 1e-6
    grad_clip: float = 1.0

    vis_freq: int = 50
    save_vis: bool = True
    routing_block_id: int = 0

    save_path: str = "/media/xd132/USER/MRZ/checkpoint/model_0813/ckpt_multi"
    log_path: str = "/media/xd132/USER/MRZ/checkpoint/model_0813/logs_multi"

    def __post_init__(self):
        assert len(self.secret_channels) == self.num_secrets
        assert len(self.secret_type_ids) == self.num_secrets

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "vis"), exist_ok=True)