"""Server-side aggregation for Paper 1 FL experiments.

Design invariants:
- Operates on CPU tensors (client state_dicts are already on CPU).
- Deterministic — order-independent, weight-aware.
- Zero-weight clients are ignored (no contribution, no error).
- Rejects mismatched state_dicts (shape/keys).
- Supports the strategies frozen in CONTRACT.md §14:
  - FedAvg:   weighted average of parameters
  - FedProx:  aggregation is identical to FedAvg; the proximal term is
              applied during local training (in client.train_local when
              proximal_mu > 0 — handled at flat_fl level, not here)
  - SCAFFOLD: control-variate correction (per-client c, c_i);
              server updates global c using weighted average of deltas.
"""
from __future__ import annotations

from typing import Dict, List

import torch


def _check_compatible(states: List[dict]) -> None:
    if not states:
        raise ValueError("client_states cannot be empty")
    first = states[0]
    keys = set(first.keys())
    for i, s in enumerate(states[1:], 1):
        if set(s.keys()) != keys:
            raise ValueError(
                f"state_dict {i} has different keys than state_dict 0"
            )
        for k in keys:
            if s[k].shape != first[k].shape:
                raise ValueError(
                    f"state_dict {i} tensor {k!r} has shape {tuple(s[k].shape)}, "
                    f"expected {tuple(first[k].shape)}"
                )


def fedavg(
    client_states: List[dict],
    client_weights: List[int],
) -> dict:
    """Weighted average of state_dicts by client sample counts.

    Args:
        client_states: list of CPU state_dicts, one per client
        client_weights: list of non-negative ints (usually n_samples)

    Returns:
        New state_dict on CPU with float32 tensors.
    """
    if len(client_states) != len(client_weights):
        raise ValueError(
            f"length mismatch: {len(client_states)} states, {len(client_weights)} weights"
        )
    _check_compatible(client_states)

    if any(w < 0 for w in client_weights):
        raise ValueError("client_weights must be non-negative")
    total_weight = sum(client_weights)
    if total_weight == 0:
        raise ValueError("sum of client_weights is 0 — cannot aggregate")

    # accumulate weighted sum
    aggregated: Dict[str, torch.Tensor] = {}
    for state, w in zip(client_states, client_weights):
        if w == 0:
            continue
        for k, v in state.items():
            contribution = v.detach().cpu().float() * (w / total_weight)
            if k in aggregated:
                aggregated[k] += contribution
            else:
                aggregated[k] = contribution

    return aggregated


def fedavg_simple(client_states: List[dict]) -> dict:
    """Uniform (unweighted) FedAvg — every client contributes equally."""
    n = len(client_states)
    return fedavg(client_states, [1] * n)


def scaffold_aggregate(
    client_states: List[dict],
    client_deltas: List[dict],
    client_weights: List[int],
    server_control: dict,
    server_lr: float = 1.0,
) -> tuple:
    """SCAFFOLD aggregation: weighted avg of parameters + control variate update.

    Args:
        client_states: per-client model state_dicts after local training
        client_deltas: per-client (c_i - c) deltas, same shapes as model
                       (often computed as new_c_i - old_c, simplified here)
        client_weights: sample counts per client
        server_control: server control variate c, same keys as model
        server_lr: server learning rate for control update

    Returns:
        (new_model_state, new_server_control)
    """
    if len(client_states) != len(client_deltas) or len(client_states) != len(client_weights):
        raise ValueError("client_states / client_deltas / client_weights lengths must match")
    _check_compatible(client_states)
    _check_compatible(client_deltas)

    new_model = fedavg(client_states, client_weights)

    # server control update: c += server_lr * weighted_avg(c_i - c)
    total = sum(client_weights)
    new_control: Dict[str, torch.Tensor] = {}
    for k in server_control.keys():
        weighted_delta = torch.zeros_like(server_control[k].detach().cpu().float())
        for delta, w in zip(client_deltas, client_weights):
            if w == 0:
                continue
            weighted_delta += delta[k].detach().cpu().float() * (w / total)
        new_control[k] = server_control[k].detach().cpu().float() + server_lr * weighted_delta

    return new_model, new_control


def aggregate(
    method: str,
    client_states: List[dict],
    client_weights: List[int],
    **kwargs,
) -> dict:
    """Dispatch on method ('fedavg' | 'fedprox' | 'scaffold')."""
    if method in ("fedavg", "fedprox"):
        # fedprox uses the same aggregation as fedavg; proximal term lives in local training
        if kwargs:
            raise ValueError(f"unexpected kwargs for {method}: {list(kwargs)}")
        return fedavg(client_states, client_weights)
    if method == "scaffold":
        client_deltas = kwargs.pop("client_deltas", None)
        server_control = kwargs.pop("server_control", None)
        server_lr = kwargs.pop("server_lr", 1.0)
        if client_deltas is None or server_control is None:
            raise ValueError("scaffold requires client_deltas= and server_control=")
        if kwargs:
            raise ValueError(f"unexpected kwargs for scaffold: {list(kwargs)}")
        new_model, _ = scaffold_aggregate(
            client_states, client_deltas, client_weights, server_control, server_lr
        )
        return new_model
    raise ValueError(
        f"Unknown aggregation method: {method!r}. "
        f"Expected 'fedavg', 'fedprox', or 'scaffold'."
    )
