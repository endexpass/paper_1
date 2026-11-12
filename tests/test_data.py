"""Unit tests for fl_core.data (Stage 1)."""
import hashlib
from pathlib import Path

import pytest
from PIL import Image
from torchvision import transforms

CIFAR_ROOT = "DATA/CIFAR100"

from fl_core.data import (
    build_cifar100,
    build_dataset,
    build_wheat,
    discover_wheat_layout,
)


# ---------------------------------------------------------------------------
# CIFAR-100
# ---------------------------------------------------------------------------
def test_build_cifar100_returns_num_classes_100():
    train, test, n = build_cifar100(root=CIFAR_ROOT, batch_size=8, num_workers=0)
    assert n == 100
    assert len(train.dataset) == 50000
    assert len(test.dataset) == 10000


def test_cifar100_batch_shape():
    train, _, _ = build_cifar100(root=CIFAR_ROOT, batch_size=4, num_workers=0)
    imgs, labels = next(iter(train))
    assert imgs.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    assert labels.min() >= 0 and labels.max() < 100


def test_cifar100_splits_are_distinct():
    """Raw CIFAR-100 train and test samples must be disjoint."""
    train, test, _ = build_cifar100(root=CIFAR_ROOT, batch_size=64, num_workers=0)

    train_sigs = {
        hashlib.sha256(train.dataset.data[i].tobytes()).hexdigest()
        for i in range(200)
    }
    test_sigs = {
        hashlib.sha256(test.dataset.data[i].tobytes()).hexdigest()
        for i in range(200)
    }
    assert train_sigs.isdisjoint(test_sigs)


def test_cifar100_idempotent():
    a = build_cifar100(root=CIFAR_ROOT, batch_size=8, num_workers=0)
    b = build_cifar100(root=CIFAR_ROOT, batch_size=8, num_workers=0)
    assert len(a[0].dataset) == len(b[0].dataset) == 50000
    assert len(a[1].dataset) == len(b[1].dataset) == 10000


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def test_build_dataset_cifar100():
    train, test, n = build_dataset(
        "cifar100", root=CIFAR_ROOT, batch_size=8, num_workers=0
    )
    assert n == 100


def test_build_dataset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset("mnist")


def test_build_dataset_wheat_without_root_raises():
    with pytest.raises(ValueError, match="requires root"):
        build_dataset("wheat")


# ---------------------------------------------------------------------------
# Wheat discovery
# ---------------------------------------------------------------------------
def _make_fake_wheat(root: Path, class_counts: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for cls, n in class_counts.items():
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.new("RGB", (32, 32), color=(i * 10 % 255, 100, 50)).save(d / f"{i}.png")
    return root


def test_discover_wheat_layout_basic(tmp_path):
    root = _make_fake_wheat(tmp_path / "wheat", {"class_a": 3, "class_b": 5})
    info = discover_wheat_layout(root)
    assert info["looks_like_imagefolder"] is True
    assert info["class_dirs"] == {"class_a": 3, "class_b": 5}
    assert info["total_image_files"] == 8


def test_discover_wheat_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_wheat_layout(tmp_path / "does_not_exist")


def test_build_wheat_discovers_num_classes(tmp_path):
    root = _make_fake_wheat(tmp_path / "wheat", {"c1": 4, "c2": 4, "c3": 4})
    train, test, n = build_wheat(root=root, batch_size=2, num_workers=0, train_split=0.5)
    assert n == 3


def test_build_wheat_batch_shape(tmp_path):
    root = _make_fake_wheat(tmp_path / "wheat", {"c1": 4, "c2": 4})
    train, _, n = build_wheat(root=root, batch_size=2, num_workers=0, train_split=0.5)
    imgs, labels = next(iter(train))
    assert imgs.shape[0] == 2
    assert imgs.shape[1:] == (3, 224, 224)
    assert labels.shape == (2,)


def test_wheat_train_transform_applied(tmp_path):
    """Training side must include RandomHorizontalFlip; test side must not."""
    root = _make_fake_wheat(tmp_path / "wheat", {"c1": 4, "c2": 4})
    train, test, _ = build_wheat(root=root, batch_size=1, num_workers=0, train_split=0.5)

    train_tf = train.dataset.dataset.transform
    test_tf = test.dataset.dataset.transform

    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_tf.transforms)
    assert not any(isinstance(t, transforms.RandomHorizontalFlip) for t in test_tf.transforms)


def test_wheat_stratified_split_preserves_all_classes(tmp_path):
    root = _make_fake_wheat(tmp_path / "wheat", {"a": 4, "b": 4, "c": 4})
    train, test, num_classes = build_wheat(
        root=root, batch_size=1, num_workers=0, train_split=0.5
    )
    assert num_classes == 3

    train_labels = {train.dataset.dataset.targets[i] for i in train.dataset.indices}
    test_labels = {test.dataset.dataset.targets[i] for i in test.dataset.indices}
    assert train_labels == {0, 1, 2}
    assert test_labels == {0, 1, 2}


def test_wheat_split_deterministic(tmp_path):
    root = _make_fake_wheat(tmp_path / "wheat", {"a": 5, "b": 5})
    a_train, a_test, _ = build_wheat(root=root, batch_size=1, num_workers=0, train_split=0.6, seed=42)
    b_train, b_test, _ = build_wheat(root=root, batch_size=1, num_workers=0, train_split=0.6, seed=42)
    assert set(a_train.dataset.indices) == set(b_train.dataset.indices)
    assert set(a_test.dataset.indices) == set(b_test.dataset.indices)


def test_build_wheat_rejects_bad_args(tmp_path):
    root = _make_fake_wheat(tmp_path / "wheat", {"a": 2})
    with pytest.raises(ValueError, match="train_split"):
        build_wheat(root=root, train_split=0.0, num_workers=0)
    with pytest.raises(ValueError, match="batch_size"):
        build_wheat(root=root, batch_size=0, num_workers=0)
    with pytest.raises(ValueError, match="num_workers"):
        build_wheat(root=root, num_workers=-1)
