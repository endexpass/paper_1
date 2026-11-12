"""Unit tests for fl_core.heterogeneity (Stage 3)."""
import json
from pathlib import Path

import numpy as np
import pytest

from fl_core.heterogeneity import (
    build_partition,
    load_partition,
    partition_dirichlet,
    partition_iid,
    partition_pathological,
    save_partition,
)


def _fake_labels(n_classes: int = 10, samples_per_class: int = 100) -> np.ndarray:
    return np.repeat(np.arange(n_classes), samples_per_class)


def _all_indices(partition: dict) -> set:
    out = set()
    for v in partition.values():
        out.update(v)
    return out


def test_iid_sizes_sum_to_total():
    labels = _fake_labels()
    p = partition_iid(labels, n_clients=10, seed=42)
    assert len(p) == 10
    assert len(_all_indices(p)) == len(labels)


def test_iid_deterministic():
    labels = _fake_labels()
    a = partition_iid(labels, n_clients=10, seed=42)
    b = partition_iid(labels, n_clients=10, seed=42)
    for k in a:
        assert a[k] == b[k]


def test_iid_different_seed_differs():
    labels = _fake_labels()
    a = partition_iid(labels, n_clients=10, seed=42)
    b = partition_iid(labels, n_clients=10, seed=999)
    assert any(a[k] != b[k] for k in a)


def test_iid_roughly_balanced_classes():
    labels = _fake_labels(n_classes=10, samples_per_class=100)
    p = partition_iid(labels, n_clients=10, seed=42)
    for client_id, idxs in p.items():
        counts = np.bincount(labels[idxs], minlength=10)
        expected = counts.sum() / 10
        deviations = np.abs(counts - expected)
        assert deviations.mean() < 4.0, f"client {client_id} mean dev too high: {counts}"


def test_dirichlet_covers_all_samples():
    labels = _fake_labels()
    p = partition_dirichlet(labels, n_clients=20, alpha=0.1, seed=42)
    assert len(_all_indices(p)) == len(labels)


def test_dirichlet_deterministic():
    labels = _fake_labels()
    a = partition_dirichlet(labels, n_clients=20, alpha=0.1, seed=42)
    b = partition_dirichlet(labels, n_clients=20, alpha=0.1, seed=42)
    for k in a:
        assert a[k] == b[k]


def test_dirichlet_low_alpha_is_skewed():
    labels = _fake_labels()
    low = partition_dirichlet(labels, n_clients=20, alpha=0.01, seed=42)
    high = partition_dirichlet(labels, n_clients=20, alpha=10.0, seed=42)

    def _max_class_share(partition):
        shares = []
        for idxs in partition.values():
            if not idxs:
                continue
            counts = np.bincount(labels[idxs])
            shares.append(counts.max() / counts.sum())
        return float(np.mean(shares))

    low_share = _max_class_share(low)
    high_share = _max_class_share(high)
    assert low_share > high_share + 0.3, (
        f"low alpha share={low_share:.2f}, high alpha share={high_share:.2f}"
    )


def test_pathological_covers_all_samples():
    labels = _fake_labels()
    p = partition_pathological(labels, n_clients=10, classes_per_client=2, seed=42)
    assert len(_all_indices(p)) == len(labels)


def test_pathological_exactly_k_classes_per_client():
    labels = _fake_labels(n_classes=10, samples_per_class=100)
    p = partition_pathological(labels, n_clients=10, classes_per_client=2, seed=42)
    for client_id, idxs in p.items():
        client_classes = set(labels[idxs].tolist())
        assert len(client_classes) <= 2, (
            f"client {client_id} sees {len(client_classes)} classes, expected <= 2"
        )


def test_build_partition_dispatches_correctly():
    labels = _fake_labels()
    p_iid = build_partition(labels, "iid", n_clients=10, seed=42)
    p_dir = build_partition(labels, "dirichlet", n_clients=10, alpha=0.1, seed=42)
    p_path = build_partition(labels, "pathological", n_clients=10, classes_per_client=2, seed=42)
    for p in (p_iid, p_dir, p_path):
        assert len(_all_indices(p)) == len(labels)


def test_build_partition_unknown_raises():
    labels = _fake_labels()
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_partition(labels, "quantum", n_clients=10, seed=42)


def test_save_load_roundtrip(tmp_path):
    labels = _fake_labels()
    p = partition_iid(labels, n_clients=10, seed=42)

    manifest_path = tmp_path / "partition_manifest.json"
    npz_path = tmp_path / "partition.npz"
    manifest = save_partition(
        p,
        manifest_path=manifest_path,
        npz_path=npz_path,
        dataset="synthetic",
        dataset_split="train",
        strategy="iid",
        n_clients=10,
        seed=42,
        n_train_samples=len(labels),
    )

    assert manifest_path.exists()
    assert npz_path.exists()
    assert "partition_hash" in manifest
    assert manifest["partition_format_version"] == 1

    loaded = load_partition(manifest_path, npz_path)
    for k in p:
        assert p[k] == loaded[k]


def test_load_detects_hash_mismatch(tmp_path):
    labels = _fake_labels()
    p = partition_iid(labels, n_clients=10, seed=42)
    manifest_path = tmp_path / "partition_manifest.json"
    npz_path = tmp_path / "partition.npz"
    save_partition(
        p,
        manifest_path=manifest_path,
        npz_path=npz_path,
        dataset="synthetic", dataset_split="train",
        strategy="iid", n_clients=10, seed=42, n_train_samples=len(labels),
    )

    m = json.loads(manifest_path.read_text())
    m["partition_hash"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    manifest_path.write_text(json.dumps(m))

    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_partition(manifest_path, npz_path)


def test_build_partition_wrong_kwarg_raises():
    labels = _fake_labels()
    with pytest.raises(ValueError, match="unexpected kwargs"):
        build_partition(labels, "iid", n_clients=10, seed=42, alpha=0.1)
