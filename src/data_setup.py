"""
data_setup.py
-------------
Simple CIFAR-10 dataloaders for DLCV 2026 Assignment 2.

Usage:
    from data_setup import get_dataloaders

    train_loader, val_loader, num_classes = get_dataloaders(
        data_fraction=0.1,   # Experiment 1: use 10% of training data
        img_size=32,         # Experiment 2: match to patch size
        batch_size=128,
    )
"""

import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# CIFAR-10 channel mean and std (pre-computed)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
NUM_CLASSES  = 10


def get_dataloaders(
    data_fraction: float = 1.0,   # Experiment 1: 0.05 / 0.1 / 0.25 / 0.5 / 1.0
    img_size:      int   = 32,    # Experiment 2: resize images (e.g. 96 for patch_size=16)
    batch_size:    int   = 128,
    num_workers:   int   = 2,
    data_dir:      str   = "./data",
    seed:          int   = 42,
):
    """
    Returns (train_loader, val_loader, num_classes).

    Args:
        data_fraction : Fraction of training data to use. 1.0 = full dataset.
        img_size      : Resize all images to (img_size x img_size).
                        Keep at 32 for patch sizes 4 and 8.
                        Use 64 or 96 for patch size 16 (so you get enough tokens).
        batch_size    : Number of samples per batch.
        num_workers   : Parallel workers for data loading.
        data_dir      : Where to download / cache CIFAR-10.
        seed          : Random seed for reproducible subset sampling.
    """

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_full = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_transform)
    val_set    = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)

    if data_fraction < 1.0:
        train_set = _get_balanced_subset(train_full, data_fraction, seed)
        print(f"Using {len(train_set)} / {len(train_full)} training samples ({data_fraction*100:.0f}%)")
    else:
        train_set = train_full

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, NUM_CLASSES


def _get_balanced_subset(dataset, fraction: float, seed: int) -> Subset:
    """
    Returns a class-balanced subset — each class contributes the same fraction.
    """
    rng = random.Random(seed)

    # Group indices by class label
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected = []
    for indices in class_indices.values():
        rng.shuffle(indices)
        k = max(1, int(len(indices) * fraction))
        selected.extend(indices[:k])

    return Subset(dataset, selected)


# Quick test — run this file directly to verify everything works
if __name__ == "__main__":
    train_loader, val_loader, num_classes = get_dataloaders(
        data_fraction=0.1,
        img_size=32,
        batch_size=128,
    )

    imgs, labels = next(iter(train_loader))
    print(f"num_classes  : {num_classes}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches  : {len(val_loader)}")
    print(f"batch shape  : {imgs.shape}")