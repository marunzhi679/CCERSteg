
import os
import torch


class Args:
    def __init__(self):
        self.seed = 42
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.data_path = os.environ.get(
            "CCER_DATA_ROOT",
            "/media/xd132/USER/ZZS/project/StegFormer/datasets"
        )
        self.batch_size = 4
        self.num_workers = 8

        self.image_size_train = 256
        self.image_size_val = 256

        self.embed_dim = 48
        self.num_experts = 6
        self.num_secrets = 3
        self.secret_channels = [3, 3, 3]
        self.secret_type_ids = [0.0, 0.0, 0.0]

        self.epochs = 1000
        self.lr = 2e-4
        self.weight_decay = 1e-5
        self.grad_clip = 1.0

        self.cosine_T0 = 1000
        self.cosine_T_mult = 1
        self.min_lr = 1e-6

        self.lambda_s1_start = 40.0
        self.lambda_s1_end = 20.0
        self.lambda_s2_start = 40.0
        self.lambda_s2_end = 20.0
        self.lambda_s3_start = 40.0
        self.lambda_s3_end = 20.0

        self.lambda_cover_start = 0.5
        self.lambda_cover_end = 14.0

        self.lambda_aux = 0.1
        self.lambda_mask = 0.02

        self.val_freq = 10

        self.save_path = os.path.join(
            os.environ.get("CCER_OUTPUT_ROOT", "./outputs"),
            "ckpt_rgb3"
        )

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(
            os.path.join(self.save_path, "models"),
            exist_ok=True
        )
