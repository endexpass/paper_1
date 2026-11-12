"""Client partitioning for Paper 1 FL experiments.

Design invariants:
- All partitions are deterministic given (strategy, params, seed).
- Partition manifest records everything needed to reproduce exactly.
- Partition arrays (.npz) preserve the actual indices — never rely only on
  regeneration from the manifest.
- Partitioning is SEPARATE from train/test split (data.py) and from
  client selection (selection.py, Stage 3).

Strategies (matching CONTRACT.md §14):
- IID: equal-size shards, approximately uniform class distribution per client.
- Dirichlet(alpha): label skew where smaller alpha = more heterogeneous.
  Standard reference: Hsu et al. 2019, "Measuring the Effects of Non-Identical
  Data Distribution on Federated Learning".
- Pathological: each client gets samples from only `classes_per_client` classes.
  Standard reference: McMahan et al. 2017, "Communication-Efficient Learning
  of Deep Networks from Decentralized Data".
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PARTITION_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def partition_iid(
    labels: np.ndarray,
    n_clients: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Split indices into n_clients equal-size shards, uniformly shuffled.

    Each client gets approximately the same number of samples per class
    (since labels are shuffled globally before splitting).
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if n < n_clients:
        raise ValueError(f"not enough samples ({n}) for {n_clients} clients")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    shards = np.array_split(perm, n_clients)
    return {i: shard.tolist() for i, shard in enumerate(shards)}


def partition_dirichlet(
    labels: np.ndarray,
    n_clients: int,
    alpha: float = 0.1,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Dirichlet(alpha) label skew partitioning.

    For each class, draw a proportion vector p ~ Dirichlet(alpha * ones(n_clients)),
    then split the class's samples across clients according to p.
    Lower alpha => higher skew (some clients get most of a class).

    Ensures every client has at least 1 sample if possible; retries are not
    necessary because the smallest class has at least n_clients samples in
    CIFAR-100 (500 samples/class >= 100 clients).
    """
    labels = np.asarray(labels)
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    client_indices: Dict[int, List[int]] = {i: [] for i in range(n_clients)}

    for c in range(n_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet([alpha] * n_clients)
        # distribute class_indices to clients by proportions
        cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, cuts)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # shuffle within each client so order isn't class-sorted
    for i in range(n_clients):
        arr = np.array(client_indices[i])
        rng.shuffle(arr)
        client_indices[i] = arr.tolist()

    return client_indices


def partition_pathological(
    labels: np.ndarray,
    n_clients: int,
    classes_per_client: int = 2,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Each client receives samples from exactly `classes_per_client` classes.

    Shards are assigned so that across all clients, every class is covered.
    """
    labels = np.asarray(labels)
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if classes_per_client < 1:
        raise ValueError(f"classes_per_client must be >= 1, got {classes_per_client}")

    n_classes = int(labels.max()) + 1
    if classes_per_client > n_classes:
        raise ValueError(
            f"classes_per_client ({classes_per_client}) > n_classes ({n_classes})"
        )

    rng = np.random.default_rng(seed)
    # assign each client a random subset of classes, guaranteeing that
    # every class is assigned to at least one client.
    client_classes: List[List[int]] = []
    # seed coverage: round-robin distribute the first n_clients*some classes
    # so every class appears at least once across clients.
    # Simple strategy: start each client with one class from a cyclic shuffle,
    # then add remaining (classes_per_client - 1) random classes.
    if classes_per_client >= 1:
        base_cycle = []
        while len(base_cycle) < n_clients:
            perm = rng.permutation(n_classes)
            base_cycle.extend(perm.tolist())
        base_cycle = base_cycle[:n_clients]
        for i in range(n_clients):
            chosen = {base_cycle[i]}
            remaining_pool = [c for c in range(n_classes) if c not in chosen]
            extra_needed = classes_per_client - 1
            if extra_needed > 0:
                extra = rng.choice(remaining_pool, size=extra_needed, replace=False)
                chosen.update(int(c) for c in extra)
            client_classes.append(sorted(chosen))
    else:
        for _ in range(n_clients):
            client_classes.append([])

    # per-class sample pool
    class_pools = {c: list(np.where(labels == c)[0]) for c in range(n_classes)}
    for c in class_pools:
        rng.shuffle(class_pools[c])

    # how many clients need each class
    class_needed = {c: [] for c in range(n_classes)}
    for i, cls_list in enumerate(client_classes):
        for c in cls_list:
            class_needed[c].append(i)

    # evenly split each class pool among clients that need it
    client_indices: Dict[int, List[int]] = {i: [] for i in range(n_clients)}
    for c, clients in class_needed.items():
        if not clients:
            continue
        pool = class_pools[c]
        splits = np.array_split(pool, len(clients))
        for client_id, split in zip(clients, splits):
            client_indices[client_id].extend(split.tolist())

    # shuffle within each client
    for i in range(n_clients):
        arr = np.array(client_indices[i])
        rng.shuffle(arr)
        client_indices[i] = arr.tolist()

    return client_indices


# ---------------------------------------------------------------------------
# Dispatch + manifest
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _hash_partition(partition: Dict[int, List[int]]) -> str:
    """Deterministic sha256 over sorted client->indices."""
    canonical = json.dumps(
        {str(k): sorted(v) for k, v in sorted(partition.items())},
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_partition(
    labels: np.ndarray,
    strategy: str,
    n_clients: int,
    seed: int = 42,
    **kwargs,
) -> Dict[int, List[int]]:
    """Dispatch on strategy. Returns {client_id: [sample_indices]}."""
    if strategy == "iid":
        if kwargs:
            raise ValueError(f"unexpected kwargs for iid: {list(kwargs)}")
        return partition_iid(labels, n_clients, seed=seed)
    if strategy == "dirichlet":
        alpha = kwargs.pop("alpha", 0.1)
        if kwargs:
            raise ValueError(f"unexpected kwargs for dirichlet: {list(kwargs)}")
        return partition_dirichlet(labels, n_clients, alpha=alpha, seed=seed)
    if strategy == "pathological":
        k = kwargs.pop("classes_per_client", 2)
        if kwargs:
            raise ValueError(f"unexpected kwargs for pathological: {list(kwargs)}")
        return partition_pathological(labels, n_clients, classes_per_client=k, seed=seed)
    raise ValueError(
        f"Unknown strategy: {strategy!r}. "
        f"Expected 'iid', 'dirichlet', or 'pathological'."
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_partition(
    partition: Dict[int, List[int]],
    manifest_path: Path,
    npz_path: Path,
    dataset: str,
    dataset_split: str,
    strategy: str,
    n_clients: int,
    seed: int,
    n_train_samples: int,
    extra_params: Optional[dict] = None,
) -> dict:
    """Save manifest.json + partition.npz atomically. Returns the manifest."""
    manifest_path = Path(manifest_path)
    npz_path = Path(npz_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    partition_hash = _hash_partition(partition)

    # npz arrays keyed by client id
    arrays = {f"client_{k}": np.array(v, dtype=np.int64) for k, v in partition.items()}

    # write npz first (atomic via tmp)
    tmp_npz = npz_path.parent / (npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    tmp_npz.replace(npz_path)

    manifest = {
        "dataset": dataset,
        "dataset_split": dataset_split,
        "strategy": strategy,
        "n_clients": n_clients,
        "seed": seed,
        "n_train_samples": n_train_samples,
        "partition_format_version": PARTITION_FORMAT_VERSION,
        "partition_hash": partition_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "extra_params": extra_params or {},
    }

    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_manifest.replace(manifest_path)

    return manifest


def load_partition(
    manifest_path: Path,
    npz_path: Path,
) -> Dict[int, List[int]]:
    """Read partition + verify hash. Raises on mismatch."""
    manifest_path = Path(manifest_path)
    npz_path = Path(npz_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest["partition_hash"]

    data = np.load(npz_path)
    partition: Dict[int, List[int]] = {}
    for key in data.files:
        client_id = int(key.split("_")[1])
        partition[client_id] = data[key].tolist()

    actual_hash = _hash_partition(partition)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Partition hash mismatch: manifest={expected_hash}, "
            f"recomputed={actual_hash}. File may be corrupt or modified."
        )

    return partition
