"""Unit tests for fl_core.selection (Stage 3b)."""
import pytest

from fl_core.selection import (
    select_clients,
    select_oort,
    select_random,
)


def test_random_returns_correct_count():
    chosen = select_random(list(range(100)), n_select=10, seed=42, round_num=0)
    assert len(chosen) == 10


def test_random_no_duplicates():
    chosen = select_random(list(range(100)), n_select=10, seed=42, round_num=0)
    assert len(set(chosen)) == len(chosen)


def test_random_deterministic_same_seed_round():
    a = select_random(list(range(100)), n_select=10, seed=42, round_num=0)
    b = select_random(list(range(100)), n_select=10, seed=42, round_num=0)
    assert a == b


def test_random_different_round_differs():
    a = select_random(list(range(100)), n_select=10, seed=42, round_num=0)
    b = select_random(list(range(100)), n_select=10, seed=42, round_num=1)
    assert a != b


def test_random_different_seed_differs():
    a = select_random(list(range(100)), n_select=10, seed=42, round_num=0)
    b = select_random(list(range(100)), n_select=10, seed=999, round_num=0)
    assert a != b


def test_random_too_many_raises():
    with pytest.raises(ValueError, match="> available"):
        select_random(list(range(10)), n_select=20, seed=42)


def test_random_zero_raises():
    with pytest.raises(ValueError, match="must be >= 1"):
        select_random(list(range(10)), n_select=0, seed=42)


def test_oort_selects_high_utility_clients():
    utilities = {i: float(i) for i in range(100)}
    chosen = select_oort(
        list(range(100)), utilities, n_select=10, seed=42, round_num=0, explore_frac=0.0
    )
    assert set(chosen) == set(range(90, 100))


def test_oort_explore_fraction_reduces_exploitation():
    utilities = {i: float(i) for i in range(100)}
    chosen_explore0 = select_oort(
        list(range(100)), utilities, n_select=10, seed=42, round_num=0, explore_frac=0.0
    )
    chosen_explore50 = select_oort(
        list(range(100)), utilities, n_select=10, seed=42, round_num=0, explore_frac=0.5
    )
    top10 = set(range(90, 100))
    assert len(set(chosen_explore0) & top10) == 10
    assert len(set(chosen_explore50) & top10) < 10


def test_oort_deterministic():
    utilities = {i: float(i) for i in range(100)}
    a = select_oort(list(range(100)), utilities, n_select=10, seed=42, round_num=0)
    b = select_oort(list(range(100)), utilities, n_select=10, seed=42, round_num=0)
    assert a == b


def test_oort_missing_utilities_default_zero():
    utilities = {0: 100.0, 1: 50.0}
    chosen = select_oort(
        list(range(10)), utilities, n_select=3, seed=42, round_num=0, explore_frac=0.0
    )
    assert 0 in chosen
    assert 1 in chosen
    assert 2 in chosen


def test_dispatch_random():
    chosen = select_clients("random", list(range(100)), n_select=10, seed=42)
    assert len(chosen) == 10


def test_dispatch_oort_requires_utilities():
    with pytest.raises(ValueError, match="requires client_utilities"):
        select_clients("oort", list(range(100)), n_select=10, seed=42)


def test_dispatch_unknown_raises():
    with pytest.raises(ValueError, match="Unknown selection method"):
        select_clients("greedy", list(range(100)), n_select=10, seed=42)
