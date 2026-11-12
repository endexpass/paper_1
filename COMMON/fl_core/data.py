"""Dataset factory for Paper 1.

Design invariants:
- Both datasets return (train_loader, test_loader, num_classes).
- Both resize to 224x224 to feed the MobileNetV2 backbone.
- ImageNet normalization (matches pretrained weights).
- Partitioning happens in heterogeneity.py (Stage 3), NOT here.
- The Wheat train/test split is deterministic (seed, train_split) and
  stratified by class. Generated once per (root, seed, train_split);
  recorded as a Stage-3 artifact via the partition manifest.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_INPUT_SIZE = 224


def _transforms(train: bool) -> transforms.Compose:
    ops = [transforms.Resize(_INPUT_SIZE)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_cifar100(
    root: str | Path = "DATA/CIFAR100",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int]:
    """Return (train_loader, test_loader, 100). Auto-downloads on first use."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.CIFAR100(
        root=str(root), train=True, download=True, transform=_transforms(train=True)
    )
    test_ds = datasets.CIFAR100(
        root=str(root), train=False, download=True, transform=_transforms(train=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, 100


def discover_wheat_layout(path: str | Path) -> dict:
    """Inspect a directory and report what it contains."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Wheat dataset not found: {root}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def _is_image(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in image_exts

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    class_dirs = {}
    for d in subdirs:
        n = sum(1 for _ in d.rglob("*") if _is_image(_))
        if n > 0:
            class_dirs[d.name] = n

    total = sum(1 for _ in root.rglob("*") if _is_image(_))
    looks_like_imagefolder = bool(class_dirs) and all(
        any(_is_image(p) for p in d.iterdir() if p.is_file())
        for d in subdirs
        if d.name in class_dirs
    )

    return {
        "root": str(root),
        "looks_like_imagefolder": looks_like_imagefolder,
        "class_dirs": class_dirs,
        "nested_dirs": [str(d) for d in subdirs],
        "total_image_files": total,
    }


def build_wheat(
    root: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
    train_split: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    """Build Wheat loaders from an ImageFolder-compatible directory.

    Stratified random split by class, deterministic given (seed, train_split).
    Two ImageFolders (train transform, eval transform) share the SAME index
    sets so training augmentation is applied to the train side only.
    """
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Wheat dataset not found: {root}")

    train_folder = datasets.ImageFolder(root=str(root), transform=_transforms(train=True))
    test_folder = datasets.ImageFolder(root=str(root), transform=_transforms(train=False))

    if train_folder.classes != test_folder.classes:
        raise RuntimeError("ImageFolder class mismatch between train and test views")

    num_classes = len(train_folder.classes)
    targets = train_folder.targets

    rng = random.Random(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for c in range(num_classes):
        cls_indices = [i for i, t in enumerate(targets) if t == c]
        rng.shuffle(cls_indices)
        n_train = int(len(cls_indices) * train_split)
        train_idx.extend(cls_indices[:n_train])
        test_idx.extend(cls_indices[n_train:])

    train_subset = Subset(train_folder, train_idx)
    test_subset = Subset(test_folder, test_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, num_classes


def build_dataset(
    name: str,
    root: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int]:
    """Dispatch on dataset name."""
    if name == "cifar100":
        return build_cifar100(
            root=root or "DATA/CIFAR100",
            batch_size=batch_size, num_workers=num_workers,
        )
    if name == "wheat":
        if root is None:
            raise ValueError(
                "build_dataset('wheat') requires root= (Kaggle dataset mount "
                "or local path). Use discover_wheat_layout(path) first to inspect."
            )
        return build_wheat(
            root=root, batch_size=batch_size, num_workers=num_workers,
        )
    raise ValueError(f"Unknown dataset: {name!r}. Expected 'cifar100' or 'wheat'.")
