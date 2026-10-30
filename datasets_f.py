
import os
import random
from pathlib import Path

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from natsort import natsorted
from torch.utils.data import Dataset


transform_train = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.ChannelShuffle(p=0.3),
    ToTensorV2(),
])

transform_test = A.Compose([
    ToTensorV2()
])


def _data_root(data_root):
    if data_root is None:
        data_root = os.environ.get(
            "CCER_DATA_ROOT",
            "/media/xd132/USER/ZZS/project/StegFormer/datasets"
        )
    return Path(data_root).expanduser()


def _image_files(folder, extensions):
    files = []
    for ext in extensions:
        files.extend(folder.glob(f"*.{ext}"))

    files = natsorted([str(p) for p in files])

    if not files:
        raise FileNotFoundError(
            f"No input images found in: {folder}"
        )

    return files


def _load_image(path, transform):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    if transform is None:
        raise ValueError(
            "A transform must be specified"
        )

    return (
        transform(image=image)["image"].float() / 255.0
    )


class COVER_Dataset(Dataset):
    def __init__(
        self,
        transforms_=None,
        mode="train",
        data_root=None
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {mode}")

        self.transform = transforms_

        folder = (
            _data_root(data_root)
            / "RGB"
            / f"RGB_{mode}"
        )

        self.files = _image_files(
            folder, ("png", "jpg")
        )

    def __getitem__(self, index):
        return _load_image(
            self.files[index], self.transform
        )

    def __len__(self):
        return len(self.files)


class SECRET_Dataset(Dataset):
    def __init__(
        self,
        transforms_=None,
        mode="train",
        data_root=None
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {mode}")

        self.transform = transforms_

        folder = (
            _data_root(data_root)
            / "RGB"
            / f"RGB_{mode}"
        )

        self.files = _image_files(
            folder, ("png", "jpg")
        )

        if len(self.files) < 2:
            raise ValueError(
                "RGB secret sampling requires "
                "at least two images per split"
            )

        self.offset = random.randint(
            1, len(self.files) - 1
        )

    def __getitem__(self, index):
        secret_index = (
            index + self.offset
        ) % len(self.files)

        return _load_image(
            self.files[secret_index],
            self.transform
        )

    def __len__(self):
        return len(self.files)


class IR_Dataset(Dataset):
    def __init__(
        self,
        transforms_=None,
        mode="train",
        data_root=None
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {mode}")

        self.transform = transforms_

        folder = (
            _data_root(data_root)
            / "IR"
            / f"IR_{mode}"
        )

        self.files = _image_files(
            folder, ("jpg",)
        )

    def __getitem__(self, index):
        return _load_image(
            self.files[index], self.transform
        )

    def __len__(self):
        return len(self.files)
