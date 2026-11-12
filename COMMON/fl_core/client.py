"""Client-side local training and evaluation for Paper 1 FL experiments.

Design invariants (CONTRACT.md §16.5):
- Standard PyTorch BatchNorm behavior — BN updates during local training.
  No freezing here. Aggregation policy for BN buffers is decided in
  flat_fl.py / hfl.py (Stage 4).
- Returns state_dict on CPU. Aggregation happens on CPU; each client
  moves its model to `device` (GPU when available) for training.
- Supports SGD and Adam optimizers to match experiment configs.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _make_optimizer(model: nn.Module, name: str, lr: float, momentum: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer: {name!r}. Expected 'sgd' or 'adam'.")


def _to_cpu_state(state: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    local_epochs: int,
    device: str = "cpu",
    optimizer_name: str = "sgd",
    lr: float = 0.01,
    momentum: float = 0.9,
    loss_fn: Optional[nn.Module] = None,
) -> dict:
    """Train `model` for `local_epochs` on `train_loader`. Returns CPU state_dict.

    If `local_epochs == 0`, returns the current state_dict without training.
    """
    if local_epochs < 0:
        raise ValueError(f"local_epochs must be >= 0, got {local_epochs}")

    if local_epochs == 0:
        return _to_cpu_state(model.state_dict())

    model.to(device)
    model.train()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, optimizer_name, lr, momentum)

    for _ in range(local_epochs):
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    return _to_cpu_state(model.state_dict())


def evaluate_local(
    model: nn.Module,
    eval_loader: DataLoader,
    device: str = "cpu",
    loss_fn: Optional[nn.Module] = None,
) -> dict:
    """Evaluate `model` on `eval_loader`. Returns {loss, accuracy, n_samples}."""
    model.to(device)
    model.eval()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "n_samples": 0}

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "n_samples": total_samples,
    }
