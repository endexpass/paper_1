"""Unit tests for fl_core.flat_fl (Stage 4c)."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_core.flat_fl import run_flat_fl


class TinyNet(nn.Module):
    def __init__(self, n_classes: int = 3, input_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def _make_client_data(n: int = 64, seed: int = 0, n_classes: int = 3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 8, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return TensorDataset(x, y)


def _loaders(n_clients: int = 5, n: int = 32):
    return {
        i: DataLoader(_make_client_data(n=n, seed=i), batch_size=16, shuffle=False, num_workers=0)
        for i in range(n_clients)
    }


def _test_loader(n: int = 64):
    return DataLoader(_make_client_data(n=n, seed=999), batch_size=32, shuffle=False, num_workers=0)


def test_n_rounds_zero_returns_initial_state():
    torch.manual_seed(0)
    model = TinyNet()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    final, history = run_flat_fl(
        model, _loaders(), _test_loader(),
        n_rounds=0, clients_per_round=2, local_epochs=1,
    )
    assert history == []
    for k in initial:
        assert torch.equal(initial[k], final[k])


def test_history_has_n_rounds_entries():
    torch.manual_seed(0)
    model = TinyNet()
    final, history = run_flat_fl(
        model, _loaders(), _test_loader(),
        n_rounds=3, clients_per_round=2, local_epochs=1,
    )
    assert len(history) == 3


def test_history_entry_keys():
    torch.manual_seed(0)
    model = TinyNet()
    _, history = run_flat_fl(
        model, _loaders(), _test_loader(),
        n_rounds=2, clients_per_round=2, local_epochs=1,
    )
    for h in history:
        assert {"round", "loss", "accuracy", "n_samples"} <= set(h.keys())


def test_round_numbers_sequential():
    torch.manual_seed(0)
    model = TinyNet()
    _, history = run_flat_fl(
        model, _loaders(), _test_loader(),
        n_rounds=4, clients_per_round=2, local_epochs=1,
    )
    assert [h["round"] for h in history] == [0, 1, 2, 3]


def test_on_round_callback_fires():
    torch.manual_seed(0)
    model = TinyNet()
    calls = []
    def cb(round_num, metrics):
        calls.append((round_num, metrics.get("round")))
    run_flat_fl(
        model, _loaders(), _test_loader(),
        n_rounds=3, clients_per_round=2, local_epochs=1,
        on_round=cb,
    )
    assert len(calls) == 3
    assert [c[0] for c in calls] == [0, 1, 2]


def test_deterministic_same_seed():
    torch.manual_seed(0)
    m1 = TinyNet()
    _, h1 = run_flat_fl(
        m1, _loaders(), _test_loader(),
        n_rounds=2, clients_per_round=2, local_epochs=1, seed=42,
    )

    torch.manual_seed(0)
    m2 = TinyNet()
    _, h2 = run_flat_fl(
        m2, _loaders(), _test_loader(),
        n_rounds=2, clients_per_round=2, local_epochs=1, seed=42,
    )

    for r1, r2 in zip(h1, h2):
        assert r1["loss"] == r2["loss"]
        assert r1["accuracy"] == r2["accuracy"]


def test_clients_per_round_validated():
    torch.manual_seed(0)
    model = TinyNet()
    with pytest.raises(ValueError, match="> available"):
        run_flat_fl(
            model, _loaders(n_clients=3), _test_loader(),
            n_rounds=1, clients_per_round=10, local_epochs=1,
        )


def test_negative_rounds_raises():
    model = TinyNet()
    with pytest.raises(ValueError, match="must be >= 0"):
        run_flat_fl(
            model, _loaders(), _test_loader(),
            n_rounds=-1, clients_per_round=2, local_epochs=1,
        )


def test_compression_runs_without_error():
    torch.manual_seed(0)
    model = TinyNet()
    final, history = run_flat_fl(
        model, _loaders(), _test_loader(),
        n_rounds=2, clients_per_round=2, local_epochs=1,
        compression_method="uniform_8bit",
    )
    assert len(history) == 2
    assert isinstance(final, dict)


def test_compression_close_to_none():
    torch.manual_seed(0)
    m1 = TinyNet()
    _, h_none = run_flat_fl(
        m1, _loaders(), _test_loader(),
        n_rounds=2, clients_per_round=2, local_epochs=1, seed=42,
        compression_method="none",
    )

    torch.manual_seed(0)
    m2 = TinyNet()
    _, h_8bit = run_flat_fl(
        m2, _loaders(), _test_loader(),
        n_rounds=2, clients_per_round=2, local_epochs=1, seed=42,
        compression_method="uniform_8bit",
    )

    for r_none, r_8bit in zip(h_none, h_8bit):
        assert abs(r_none["accuracy"] - r_8bit["accuracy"]) < 0.15


def test_final_state_keys_match_initial():
    torch.manual_seed(0)
    model = TinyNet()
    initial_keys = set(model.state_dict().keys())
    final, _ = run_flat_fl(
        model, _loaders(), _test_loader(),
        n_rounds=2, clients_per_round=2, local_epochs=1,
    )
    assert set(final.keys()) == initial_keys
