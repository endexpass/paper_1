"""Flat Federated Learning loop for Paper 1.

Design invariants:
- Single global model, aggregated at server after each round.
- Uses selection.select_clients for client sampling.
- Uses client.train_local for local updates.
- Uses aggregation.fedavg for the weighted average.
- Uses compression.compress/decompress for the E4 compression condition.
- Deterministic given (seed, selection_method, all params).
- Round metrics recorded after each round: {round, loss, accuracy, n_samples}.
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


def run_flat_fl(
    model: nn.Module,
    client_loaders: Dict[int, DataLoader],
    test_loader: DataLoader,
    n_rounds: int,
    clients_per_round: int,
    local_epochs: int,
    selection_method: str = "random",
    aggregation_method: str = "fedavg",
    compression_method: str = "none",
    seed: int = 42,
    device: str = "cpu",
    lr: float = 0.01,
    optimizer_name: str = "sgd",
    client_utilities: Optional[Dict[int, float]] = None,
    on_round: Optional[Callable[[int, dict], None]] = None,
) -> Tuple[dict, List[dict]]:
    """Run Flat FL end-to-end.

    Args:
        model: initial global model (will be mutated)
        client_loaders: {client_id -> DataLoader of that client's data}
        test_loader: global test set (evaluated each round)
        n_rounds: number of communication rounds
        clients_per_round: number selected each round
        local_epochs: local training epochs per round
        selection_method: "random" | "oort"
        aggregation_method: "fedavg" | "fedprox" | "scaffold"
        compression_method: "none" | "uniform_8bit"
        seed: base RNG seed
        device: torch device
        lr, optimizer_name: local training settings
        client_utilities: for oort selection
        on_round: optional callback(round_num, metrics) after each round

    Returns:
        (final_state_dict, history) where history is a list of dicts:
        [{"round": 0, "loss": ..., "accuracy": ..., "n_samples": ...}, ...]
    """
    if n_rounds < 0:
        raise ValueError(f"n_rounds must be >= 0, got {n_rounds}")
    if n_rounds == 0:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, []

    all_client_ids = sorted(client_loaders.keys())
    if not all_client_ids:
        raise ValueError("client_loaders is empty")

    history: List[dict] = []
    for round_num in range(n_rounds):
        # --- 1. select clients ---
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

        # --- 2. local training ---
        client_states: List[dict] = []
        client_weights: List[int] = []
        for cid in selected:
            loader = client_loaders[cid]
            state = train_local(
                model=model,
                train_loader=loader,
                local_epochs=local_epochs,
                device=device,
                optimizer_name=optimizer_name,
                lr=lr,
            )
            # Optionally compress then decompress (E4 condition)
            if compression_method != "none":
                payload = compress(state, compression_method)
                state = decompress(payload, compression_method)
            client_states.append(state)
            # weight = client's sample count this round
            client_weights.append(len(loader.dataset))

        # --- 3. aggregate ---
        new_state = aggregate(
            aggregation_method, client_states, client_weights,
        )
        model.load_state_dict(new_state)

        # --- 4. evaluate ---
        metrics = evaluate_local(model, test_loader, device=device)
        metrics["round"] = round_num
        history.append(metrics)
        if on_round is not None:
            on_round(round_num, metrics)

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return final_state, history
