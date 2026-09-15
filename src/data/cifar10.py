"""CIFAR-10 image-input pipelines for LogicTreeNet."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

CIFAR10_SMALL_THRESHOLDS: tuple[float, ...] = tuple((i + 1) / 4 for i in range(3))
DEFAULT_LARGE_THRESHOLDS: tuple[float, ...] = tuple((i + 1) / 32 for i in range(31))
CIFAR10_LOGICTREE_SMALL_CHANNELS: int = 3 * len(CIFAR10_SMALL_THRESHOLDS)  # 9
CIFAR10_LOGICTREE_LARGE_CHANNELS: int = 3 * len(DEFAULT_LARGE_THRESHOLDS)  # 93
CIFAR10_LOGICTREE_LARGE_BITPLANE_CHANNELS: int = 15


def _threshold_image(thresholds: tuple[float, ...]):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: torch.cat([(x > t).float() for t in thresholds], dim=0)
            ),
        ]
    )


def _bitplane5_image():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: torch.cat(
                    [
                        (((torch.round(x.clamp(0.0, 1.0) * 31.0).long() >> bit) & 1).float()) # 计算二进制 >> bit：右移 bit 位，获得对应的二进制位。 & 1：与 1 做按位与，得到该比特位（0 或 1）。
                        for bit in range(4, -1, -1)
                    ],
                    dim=0,
                )
            ),
        ]
    )

def get_cifar10_logictree_input_channels(dcfg: dict) -> int:
    mode = str(dcfg.get("cifar10_input_mode", "small")).strip().lower().replace("-", "_")
    if mode == "small":
        return CIFAR10_LOGICTREE_SMALL_CHANNELS
    if mode == "large_threshold31":
        return CIFAR10_LOGICTREE_LARGE_CHANNELS
    if mode == "large_bitplane5":
        return CIFAR10_LOGICTREE_LARGE_BITPLANE_CHANNELS
    raise ValueError(
        "Unknown data.cifar10_input_mode "
        f"{mode!r}; expected 'small', 'large_threshold31', or 'large_bitplane5'."
    )


Cifar10InputMode = Literal["small", "large_threshold31", "large_bitplane5"]


class DualInputDataset(Dataset):
    def __init__(self, student_ds: Dataset, teacher_ds: Dataset):
        self.student_ds = student_ds
        self.teacher_ds = teacher_ds
        if len(self.student_ds) != len(self.teacher_ds):
            raise ValueError("student_ds and teacher_ds must have the same length")

    def __len__(self) -> int:
        return len(self.student_ds)

    def __getitem__(self, idx: int):
        x_student, y_student = self.student_ds[idx]
        x_teacher, y_teacher = self.teacher_ds[idx]
        if int(y_student) != int(y_teacher):
            raise RuntimeError("Label mismatch between student and teacher datasets")
        return (x_student, x_teacher), y_student


def get_cifar10_loaders(
    batch_size: int,
    data_dir: str | Path = "data",
    val_split: int = 5000,
    data_seed: int = 42,
    num_workers: int = 2,
    input_mode: Cifar10InputMode = "small",
    for_teacher: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val / test loaders for CIFAR-10 image input modes."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    input_mode = str(input_mode).strip().lower().replace("-", "_")
    if input_mode == "small":
        tfm = _threshold_image(CIFAR10_SMALL_THRESHOLDS)
    elif input_mode == "large_threshold31":
        tfm = _threshold_image(DEFAULT_LARGE_THRESHOLDS)
    elif input_mode == "large_bitplane5":
        tfm = _bitplane5_image()
    else:
        raise ValueError(
            f"Unknown input_mode: {input_mode!r}; expected 'small', "
            "'large_threshold31', or 'large_bitplane5'."
        )
   

    train_full_for_split = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=tfm)
    train_full_student = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=tfm)
    val_full_student = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=tfm)
    test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=tfm)

    if val_split <= 0 or val_split >= len(train_full_for_split):
        raise ValueError("val_split must be in (0, len(train_full))")

    gen = torch.Generator().manual_seed(int(data_seed))
    train_split, val_split_dataset = random_split(
        train_full_for_split,
        [len(train_full_for_split) - val_split, val_split],
        generator=gen,
    )
    if for_teacher:
        train_full_teacher = datasets.CIFAR10(
            root=str(data_dir), train=True, download=False, transform=transforms.ToTensor()
        )
        train_dual = DualInputDataset(train_full_student, train_full_teacher)
        train_set = Subset(train_dual, train_split.indices)
    else:
        train_set = Subset(train_full_student, train_split.indices)
    val_set = Subset(val_full_student, val_split_dataset.indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
