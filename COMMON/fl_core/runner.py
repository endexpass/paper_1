"""Experiment runner for Paper 1 FL experiments.

Orchestrates the paired Flat+HFL run for one (exp_id, seed) using the
frozen EXPERIMENT_SPECS. Enforces authorization and completion gates
before any training begins.

CLI:
    python -m fl_core.runner --exp E0 --seed 42 [--resume] [--device cpu]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fl_core.checkpoint import save_checkpoint, save_result
from fl_core.data import build_cifar100
from fl_core.experiment_specs import (
    EXPERIMENT_SPECS,
    authorize_execution,
    check_completion_gates,
    get_experiment_spec,
)
from fl_core.flat_fl import run_flat_fl
from fl_core.heterogeneity import build_partition, save_partition
from fl_core.hfl import run_hfl
from fl_core.model import build_model


ARCHIVE_DIR = Path("ARCHIVE")
RESULTS_ROOT = Path("RESULTS")
CHECKPOINTS_ROOT = Path("CHECKPOINTS")
AUDIT_PATH = Path("AUDIT/run_registry.csv")
PARTITIONS_ROOT = Path("PARTITIONS")


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _capture_env() -> dict:
    import platform as _p
    import numpy as _np
    env = {
        "platform": (
            "colab" if __import__("os").environ.get("COLAB_GPU")
            else "kaggle" if __import__("os").environ.get("KAGGLE_KERNEL_RUN_TYPE")
            else "local"
        ),
        "python": _p.python_version(),
        "torch": torch.__version__,
        "numpy": _np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torchvision as _tv
        env["torchvision"] = _tv.__version__
    except Exception:
        pass
    try:
        import pandas as _pd
        env["pandas"] = _pd.__version__
    except Exception:
        pass
    return env


def _make_client_loaders(
    train_dataset,
    partition: Dict[int, List[int]],
    batch_size: int = 64,
) -> Dict[int, DataLoader]:
    loaders: Dict[int, DataLoader] = {}
    for cid, indices in partition.items():
        subset = Subset(train_dataset, indices)
        loaders[cid] = DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=0,
        )
    return loaders


def _extract_labels(dataset):
    import numpy as np
    if isinstance(dataset, Subset):
        parent = _extract_labels(dataset.dataset)
        return parent[dataset.indices]
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    raise ValueError(f"cannot extract labels from {type(dataset).__name__}")


def _load_or_build_partition(
    spec: dict,
    dataset_name: str,
    train_dataset,
    seed: int,
    n_clients: int = 10,
) -> Dict[int, List[int]]:
    part_dir = PARTITIONS_ROOT / dataset_name / f"n{n_clients}_seed{seed}"
    manifest_path = part_dir / "partition_manifest.json"
    npz_path = part_dir / "partition.npz"

    if manifest_path.exists() and npz_path.exists():
        from fl_core.heterogeneity import load_partition
        return load_partition(manifest_path, npz_path)

    strategy = spec.get("partition_type", "iid")
    extra = {}
    if strategy == "dirichlet":
        extra["alpha"] = float(spec.get("alpha", 0.1))
    elif strategy == "pathological":
        extra["classes_per_client"] = int(spec.get("classes_per_client", 2))

    labels = _extract_labels(train_dataset)
    partition = build_partition(
        labels=labels, strategy=strategy,
        n_clients=n_clients, seed=seed, **extra,
    )
    save_partition(
        partition=partition,
        manifest_path=manifest_path,
        npz_path=npz_path,
        dataset=dataset_name,
        dataset_split="train",
        strategy=strategy,
        n_clients=n_clients,
        seed=seed,
        n_train_samples=len(labels),
        extra_params=extra,
    )
    return partition

def run_experiment(
    exp_id: str,
    seed: int,
    dataset_name: str = "cifar100",
    n_clients: int = 10,
    clients_per_round: int = 5,
    n_edges: int = 2,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict:
    """Run the paired Flat+HFL experiment for one (exp_id, seed).

    Steps:
      1. authorize_execution (spec must be frozen+approved+hash-matched)
      2. check_completion_gates (E12/E13 prerequisites)
      3. build model + dataset
      4. load or generate partition
      5. run Flat FL
      6. restore initial model state
      7. run HFL on same partition + same initial state
      8. write RESULTS/ + AUDIT/run_registry.csv + checkpoints
    """
    # --- 1. authorization ---
    authorize_execution(exp_id, ARCHIVE_DIR)

    # --- 2. completion gates ---
    check_completion_gates(
        exp_id, dataset_name,
        results_dir=RESULTS_ROOT,
        logs_dir=Path("LOGS"),
        archive_dir=ARCHIVE_DIR,
    )

    # --- 3. spec + dataset ---
    spec = get_experiment_spec(exp_id)
    if dataset_name == "cifar100":
        train_loader, test_loader, num_classes = build_cifar100(batch_size=batch_size)
        train_dataset = train_loader.dataset
    else:
        raise NotImplementedError(f"dataset {dataset_name!r} not wired for E0; use cifar100")

    # --- 4. partition + client loaders ---
    partition = _load_or_build_partition(
        spec=spec, dataset_name=dataset_name,
        train_dataset=train_dataset, seed=seed, n_clients=n_clients,
    )
    client_loaders = _make_client_loaders(train_dataset, partition, batch_size=batch_size)

    # --- 5. edges: round-robin over clients ---
    clients_by_edge: Dict[int, List[int]] = {e: [] for e in range(n_edges)}
    for cid in sorted(client_loaders.keys()):
        clients_by_edge[cid % n_edges].append(cid)

    # --- 6. model + initial snapshot ---
    model = build_model(num_classes=num_classes, pretrained=True)
    initial_state = copy.deepcopy(model.state_dict())

    # --- 7. run Flat FL ---
    n_rounds = int(spec["global_rounds"])
    local_epochs = int(spec["local_epochs"])

    flat_state, flat_history = run_flat_fl(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        n_rounds=n_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        selection_method=spec.get("client_selection", "random"),
        aggregation_method=spec.get("aggregation", "fedavg"),
        compression_method=spec.get("compression", "none"),
        seed=seed,
        device=device,
    )

    # --- 8. restore initial state for HFL ---
    model.load_state_dict(initial_state)

    hfl_state, hfl_history = run_hfl(
        model=model,
        client_loaders=client_loaders,
        clients_by_edge=clients_by_edge,
        test_loader=test_loader,
        n_rounds=n_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        edge_aggregation_method="fedavg",
        server_aggregation_method=spec.get("aggregation", "fedavg"),
        compression_method=spec.get("compression", "none"),
        selection_method=spec.get("client_selection", "random"),
        seed=seed,
        device=device,
    )

    # --- 9. write results ---
    result_dir = RESULTS_ROOT / dataset_name.upper() / exp_id / f"seed_{seed}"
    flat_dir = result_dir / "FLAT"
    hfl_dir = result_dir / "HFL"
    flat_dir.mkdir(parents=True, exist_ok=True)
    hfl_dir.mkdir(parents=True, exist_ok=True)

    spec_hash = spec.get("_spec_hash", "unknown")
    code_commit = _git_commit()
    env = _capture_env()
    now_iso = datetime.now(timezone.utc).isoformat()

    flat_result = {
        "exp_id": exp_id, "seed": seed, "architecture": "FLAT",
        "dataset": dataset_name, "n_rounds": n_rounds, "local_epochs": local_epochs,
        "history": flat_history, "final_metrics": flat_history[-1] if flat_history else {},
        "environment": env, "code_commit": code_commit, "spec_hash": spec_hash,
        "finished_utc": now_iso,
    }
    hfl_result = {
        "exp_id": exp_id, "seed": seed, "architecture": "HFL",
        "dataset": dataset_name, "n_rounds": n_rounds, "local_epochs": local_epochs,
        "n_edges": n_edges, "history": hfl_history,
        "final_metrics": hfl_history[-1] if hfl_history else {},
        "environment": env, "code_commit": code_commit, "spec_hash": spec_hash,
        "finished_utc": now_iso,
    }

    save_result(flat_result, flat_dir / "result.json")
    save_result(hfl_result, hfl_dir / "result.json")

    summary = {
        "exp_id": exp_id, "seed": seed, "dataset": dataset_name,
        "flat_final_accuracy": flat_result["final_metrics"].get("accuracy"),
        "hfl_final_accuracy": hfl_result["final_metrics"].get("accuracy"),
        "flat_final_loss": flat_result["final_metrics"].get("loss"),
        "hfl_final_loss": hfl_result["final_metrics"].get("loss"),
        "generated_utc": now_iso,
    }
    save_result(summary, result_dir / "aggregate_summary.json")

    # --- 10. checkpoints ---
    ckpt_dir = CHECKPOINTS_ROOT / dataset_name.upper() / exp_id / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(flat_state, ckpt_dir / "FLAT_final.pt", round_num=n_rounds,
                    metadata={"exp_id": exp_id, "seed": seed, "arch": "FLAT"})
    save_checkpoint(hfl_state, ckpt_dir / "HFL_final.pt", round_num=n_rounds,
                    metadata={"exp_id": exp_id, "seed": seed, "arch": "HFL"})

    # --- 11. audit registry ---
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_row = (
        f"{exp_id}_{seed},{exp_id},{dataset_name.upper()},FLAT+HFL,local,{seed},"
        f"{spec.get('aggregation', 'fedavg')},{code_commit},{spec_hash},,"
        f"{now_iso},{now_iso},completed,RESULTS/{dataset_name.upper()}/{exp_id}/seed_{seed}\n"
    )
    if AUDIT_PATH.exists():
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(new_row)
    else:
        header = "run_id,exp_id,dataset,architecture,platform,seed,variant,code_commit,spec_hash,partition_hash,started_utc,finished_utc,status,metrics_file\n"
        with open(AUDIT_PATH, "w", encoding="utf-8") as f:
            f.write(header + new_row)

    return {
        "exp_id": exp_id, "seed": seed,
        "result_dir": str(result_dir),
        "flat": flat_result, "hfl": hfl_result,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fl_core.runner",
        description="Run a Paper 1 FL experiment (paired Flat + HFL).",
    )
    parser.add_argument("--exp", required=True, help="Experiment ID (E0..E13)")
    parser.add_argument("--seed", type=int, required=True, help="RNG seed")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-clients", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=5)
    parser.add_argument("--n-edges", type=int, default=2)
    args = parser.parse_args(argv)

    print(f"[runner] exp={args.exp} seed={args.seed} device={args.device}")
    result = run_experiment(
        exp_id=args.exp, seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        n_clients=args.n_clients,
        clients_per_round=args.clients_per_round,
        n_edges=args.n_edges,
    )
    print(f"[runner] done. results at {result['result_dir']}")
    print(f"[runner] FLAT final accuracy: {result['flat']['final_metrics'].get('accuracy')}")
    print(f"[runner] HFL  final accuracy: {result['hfl']['final_metrics'].get('accuracy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
