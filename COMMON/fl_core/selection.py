"""Client selection for Paper 1 FL experiments.

Design invariants:
- Deterministic given (seed, round_num).
- Different rounds select different subsets (round_num salts the seed).
- Never selects more clients than exist.
- Returns client IDs in a consistent order (sorted) for reproducibility.

Methods (matching CONTRACT.md §14):
- random: uniform sample of n_select from n_total.
- oort:   guided participant selection. Top (1-explore_frac) chosen by
          utility (typically estimated loss/latency); remaining
          explore_frac*n_select chosen uniformly at random.
          Reference: Lai et al. 2021, "Oort: Efficient Federated Learning
          via Guided Participant Selection", OSDI.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _round_seed(seed: int, round_num: int) -> int:
    """Combine base seed and round_num deterministically."""
    # simple, non-cryptographic mix — enough for reproducibility
    return (int(seed) * 1_000_003 + int(round_num) * 31) & 0x7FFFFFFF


def select_random(
    client_ids: List[int],
    n_select: int,
    seed: int = 42,
    round_num: int = 0,
) -> List[int]:
    """Uniform random selection of n_select clients from client_ids."""
    if n_select < 1:
        raise ValueError(f"n_select must be >= 1, got {n_select}")
    if n_select > len(client_ids):
        raise ValueError(
            f"n_select ({n_select}) > available clients ({len(client_ids)})"
        )
    rng = np.random.default_rng(_round_seed(seed, round_num))
    chosen = rng.choice(client_ids, size=n_select, replace=False)
    return sorted(int(c) for c in chosen)


def select_oort(
    client_ids: List[int],
    client_utilities: Dict[int, float],
    n_select: int,
    seed: int = 42,
    round_num: int = 0,
    explore_frac: float = 0.1,
) -> List[int]:
    """Oort-style guided selection: exploit top utilities, explore randomly.

    Args:
        client_ids: all available client IDs
        client_utilities: {client_id: utility_score}; higher = better.
                          Missing clients default to 0.0 (lowest priority).
        n_select: number to select
        seed: RNG seed
        round_num: current round (salt for seed)
        explore_frac: fraction of n_select to pick uniformly at random
                      (default 0.1 = 10% exploration)
    """
    if n_select < 1:
        raise ValueError(f"n_select must be >= 1, got {n_select}")
    if n_select > len(client_ids):
        raise ValueError(
            f"n_select ({n_select}) > available clients ({len(client_ids)})"
        )
    if not 0 <= explore_frac <= 1:
        raise ValueError(f"explore_frac must be in [0, 1], got {explore_frac}")

    n_explore = int(round(n_select * explore_frac))
    n_exploit = n_select - n_explore

    rng = np.random.default_rng(_round_seed(seed, round_num))

    # Exploit phase: sort by utility (descending), tie-break by client_id
    ranked = sorted(
        client_ids,
        key=lambda c: (-client_utilities.get(c, 0.0), c),
    )
    exploit = ranked[:n_exploit]

    # Explore phase: uniform random from the remaining pool
    remaining = [c for c in client_ids if c not in set(exploit)]
    if n_explore > 0 and remaining:
        chosen = rng.choice(remaining, size=min(n_explore, len(remaining)), replace=False)
        explore = [int(c) for c in chosen]
    else:
        explore = []

    return sorted(exploit + explore)


def select_clients(
    method: str,
    client_ids: List[int],
    n_select: int,
    seed: int = 42,
    round_num: int = 0,
    **kwargs,
) -> List[int]:
    """Dispatch on method ('random' | 'oort')."""
    if method == "random":
        if kwargs:
            raise ValueError(f"unexpected kwargs for random: {list(kwargs)}")
        return select_random(client_ids, n_select, seed=seed, round_num=round_num)
    if method == "oort":
        client_utilities = kwargs.pop("client_utilities", None)
        if client_utilities is None:
            raise ValueError("oort requires client_utilities=...")
        explore_frac = kwargs.pop("explore_frac", 0.1)
        if kwargs:
            raise ValueError(f"unexpected kwargs for oort: {list(kwargs)}")
        return select_oort(
            client_ids, client_utilities, n_select,
            seed=seed, round_num=round_num, explore_frac=explore_frac,
        )
    raise ValueError(
        f"Unknown selection method: {method!r}. Expected 'random' or 'oort'."
    )
