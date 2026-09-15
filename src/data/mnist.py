"""MNIST image loaders for LogicTreeNet."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def _binarize_image(threshold: float):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x >= threshold).float()),
        ]
    )


def get_mnist_loaders(
    batch_size: int,
    data_dir: str | Path = "data",
    binarize_threshold: float = 0.5,
    val_split: int = 5000,
    data_seed: int = 42,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val (from train) / test loaders."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tfm = _binarize_image(binarize_threshold)

    train_full_for_split = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    train_full = datasets.MNIST(root=str(data_dir), train=True, download=False, transform=tfm)
    val_full = datasets.MNIST(root=str(data_dir), train=True, download=False, transform=tfm)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tfm)

    if val_split <= 0 or val_split >= len(train_full_for_split):
        raise ValueError("val_split must be in (0, len(train_full))")

    gen = torch.Generator().manual_seed(int(data_seed))
    train_split, val_split_dataset = random_split(
        train_full_for_split,
        [len(train_full_for_split) - val_split, val_split],
        generator=gen,
    )
    train_set = Subset(train_full, train_split.indices)
    val_set = Subset(val_full, val_split_dataset.indices)

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
