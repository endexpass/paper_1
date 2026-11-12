"""Hierarchical Federated Learning (HFL) loop for Paper 1.

Design invariants:
- Clients are grouped into edges. Each round:
    1. Global client selection (as in Flat FL)
    2. Per-edge local aggregation of selected clients on that edge
    3. Server aggregation of edge models
- Only `n_edges_with_selected_clients` state_dicts reach the server per
  round — not `n_clients_per_round`. This is the paper's efficiency claim.
- Round metrics include `n_states_to_server` and
  `communication_bytes_to_server` for bandwidth accounting.
- Deterministic given (seed, selection_method, all params).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_core.aggregation import aggregate
from fl_core.client import evaluate_local, train_local
from fl_core.compression import compress, decompress
from fl_core.selection import select_clients


def _state_dict_bytes(state: dict) -> int:
    return sum(v.numel() * v.element_size() for v in state.values())


def run_hfl(
    model: nn.Module,
    client_loaders: Dict[int, DataLoader],
    clients_by_edge: Dict[int, List[int]],
    test_loader: DataLoader,
    n_rounds: int,
    clients_per_round: int,
    local_epochs: int,
    edge_aggregation_method: str = "fedavg",
    server_aggregation_method: str = "fedavg",
    compression_method: str = "none",
    selection_method: str = "random",
    seed: int = 42,
    device: str = "cpu",
    lr: float = 0.01,
    optimizer_name: str = "sgd",
    client_utilities: Optional[Dict[int, float]] = None,
    on_round: Optional[Callable[[int, dict], None]] = None,
) -> Tuple[dict, List[dict]]:
    """Run HFL end-to-end. Returns (final_state_dict, history)."""
    if n_rounds < 0:
        raise ValueError(f"n_rounds must be >= 0, got {n_rounds}")
    if n_rounds == 0:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, []

    all_client_ids = sorted(client_loaders.keys())
    if not all_client_ids:
        raise ValueError("client_loaders is empty")

    # build client -> edge map; reject clients not in any edge
    client_to_edge: Dict[int, int] = {}
    for edge_id, cids in clients_by_edge.items():
        for cid in cids:
            if cid in client_to_edge:
                raise ValueError(f"client {cid} assigned to multiple edges")
            client_to_edge[cid] = edge_id
    for cid in all_client_ids:
        if cid not in client_to_edge:
            raise ValueError(
                f"client {cid} is in client_loaders but not in clients_by_edge"
            )

    history: List[dict] = []
    for round_num in range(n_rounds):
        # --- 1. global client selection ---
        if selection_method == "oort":
            if client_utilities is None:
                raise ValueError("oort selection requires client_utilities=...")
            selected = select_clients(
                "oort", all_client_ids, clients_per_round,
                seed=seed, round_num=round_num,
                client_utilities=client_utilities,
            )
        else:
            selected = select_clients(
                selection_method, all_client_ids, clients_per_round,
                seed=seed, round_num=round_num,
            )

        # --- 2. group selected clients by edge ---
        selected_by_edge: Dict[int, List[int]] = {}
        for cid in selected:
            e = client_to_edge[cid]
            selected_by_edge.setdefault(e, []).append(cid)

        # --- 3. per-edge local aggregation ---
        edge_states: List[dict] = []
        edge_weights: List[int] = []
        for edge_id, cids in selected_by_edge.items():
            client_states: List[dict] = []
            client_weights: List[int] = []
            for cid in cids:
                loader = client_loaders[cid]
                state = train_local(
                    model=model,
                    train_loader=loader,
                    local_epochs=local_epochs,
                    device=device,
                    optimizer_name=optimizer_name,
                    lr=lr,
                )
                if compression_method != "none":
                    payload = compress(state, compression_method)
                    state = decompress(payload, compression_method)
                client_states.append(state)
                client_weights.append(len(loader.dataset))
            edge_state = aggregate(edge_aggregation_method, client_states, client_weights)
            edge_states.append(edge_state)
            edge_weights.append(sum(client_weights))

        # --- 4. server aggregation of edge models ---
        new_state = aggregate(server_aggregation_method, edge_states, edge_weights)
        model.load_state_dict(new_state)

        # --- 5. evaluate + record bandwidth ---
        metrics = evaluate_local(model, test_loader, device=device)
        metrics["round"] = round_num
        metrics["n_client_updates"] = len(selected)
        metrics["n_states_to_server"] = len(edge_states)
        metrics["n_edge_updates"] = len(selected_by_edge)
        metrics["communication_bytes_to_server"] = len(edge_states) * _state_dict_bytes(edge_states[0])
        history.append(metrics)
        if on_round is not None:
            on_round(round_num, metrics)

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return final_state, history
