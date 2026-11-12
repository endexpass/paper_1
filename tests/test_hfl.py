"""Unit tests for fl_core.hfl (Stage 4d)."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_core.hfl import run_hfl


class TinyNet(nn.Module):
    def __init__(self, n_classes: int = 3, input_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def _make_data(n: int = 64, seed: int = 0, n_classes: int = 3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 8, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return TensorDataset(x, y)


def _loaders(n_clients: int = 6, n: int = 32):
    return {
        i: DataLoader(_make_data(n=n, seed=i), batch_size=16, shuffle=False, num_workers=0)
        for i in range(n_clients)
    }


def _edges(n_clients: int = 6, edges: int = 2):
    out = {e: [] for e in range(edges)}
    for cid in range(n_clients):
        out[cid % edges].append(cid)
    return out


def _test_loader(n: int = 64):
    return DataLoader(_make_data(n=n, seed=999), batch_size=32, shuffle=False, num_workers=0)


def _state_bytes(model):
    return sum(v.numel() * v.element_size() for v in model.state_dict().values())


def test_hfl_zero_rounds_returns_initial_state():
    torch.manual_seed(0)
    model = TinyNet()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    final, history = run_hfl(
        model, _loaders(), _edges(), _test_loader(),
        n_rounds=0, clients_per_round=2, local_epochs=1,
    )
    assert history == []
    for k in initial:
        assert torch.equal(initial[k], final[k])


def test_hfl_history_has_n_rounds():
    torch.manual_seed(0)
    model = TinyNet()
    _, history = run_hfl(
        model, _loaders(), _edges(), _test_loader(),
        n_rounds=3, clients_per_round=4, local_epochs=1,
    )
    assert len(history) == 3


def test_hfl_history_keys():
    torch.manual_seed(0)
    model = TinyNet()
    _, history = run_hfl(
        model, _loaders(), _edges(), _test_loader(),
        n_rounds=2, clients_per_round=4, local_epochs=1,
    )
    for h in history:
        assert {"round", "loss", "accuracy", "n_samples",
                "n_client_updates", "n_states_to_server",
                "n_edge_updates", "communication_bytes_to_server"} <= set(h.keys())


def test_hfl_n_states_to_server_less_than_flat():
    """The key paper claim: HFL reduces server-side updates vs Flat."""
    torch.manual_seed(0)
    model = TinyNet()
    _, history = run_hfl(
        model, _loaders(6), _edges(6, 2), _test_loader(),
        n_rounds=1, clients_per_round=4, local_epochs=1, seed=42,
    )
    h = history[0]
    assert h["n_client_updates"] == 4
    assert h["n_states_to_server"] <= 2
    assert h["n_states_to_server"] < h["n_client_updates"]


def test_hfl_communication_bytes_reduced():
    torch.manual_seed(0)
    model = TinyNet()
    _, history = run_hfl(
        model, _loaders(6), _edges(6, 2), _test_loader(),
        n_rounds=1, clients_per_round=4, local_epochs=1,
    )
    h = history[0]
    client_state_bytes = _state_bytes(model)
    flat_bytes = h["n_client_updates"] * client_state_bytes
    hfl_bytes = h["communication_bytes_to_server"]
    assert hfl_bytes < flat_bytes


def test_hfl_deterministic():
    torch.manual_seed(0)
    m1 = TinyNet()
    _, h1 = run_hfl(m1, _loaders(), _edges(), _test_loader(),
                    n_rounds=2, clients_per_round=4, local_epochs=1, seed=42)

    torch.manual_seed(0)
    m2 = TinyNet()
    _, h2 = run_hfl(m2, _loaders(), _edges(), _test_loader(),
                    n_rounds=2, clients_per_round=4, local_epochs=1, seed=42)

    for r1, r2 in zip(h1, h2):
        assert r1["loss"] == r2["loss"]
        assert r1["accuracy"] == r2["accuracy"]


def test_hfl_on_round_callback():
    torch.manual_seed(0)
    model = TinyNet()
    calls = []
    def cb(round_num, metrics):
        calls.append(round_num)
    run_hfl(model, _loaders(), _edges(), _test_loader(),
            n_rounds=3, clients_per_round=4, local_epochs=1,
            on_round=cb)
    assert calls == [0, 1, 2]


def test_hfl_client_not_in_edge_raises():
    torch.manual_seed(0)
    model = TinyNet()
    edges = {0: [0, 1], 1: [2, 3]}
    with pytest.raises(ValueError, match="not in clients_by_edge"):
        run_hfl(model, _loaders(6), edges, _test_loader(),
                n_rounds=1, clients_per_round=2, local_epochs=1)


def test_hfl_duplicate_edge_assignment_raises():
    torch.manual_seed(0)
    model = TinyNet()
    edges = {0: [0, 1], 1: [1, 2]}
    with pytest.raises(ValueError, match="multiple edges"):
        run_hfl(model, _loaders(3), edges, _test_loader(),
                n_rounds=1, clients_per_round=2, local_epochs=1)


def test_hfl_compression_runs():
    torch.manual_seed(0)
    model = TinyNet()
    final, history = run_hfl(
        model, _loaders(), _edges(), _test_loader(),
        n_rounds=2, clients_per_round=4, local_epochs=1,
        compression_method="uniform_8bit",
    )
    assert len(history) == 2


def test_hfl_final_state_shape():
    torch.manual_seed(0)
    model = TinyNet()
    initial_keys = set(model.state_dict().keys())
    final, _ = run_hfl(
        model, _loaders(), _edges(), _test_loader(),
        n_rounds=2, clients_per_round=4, local_epochs=1,
    )
    assert set(final.keys()) == initial_keys


def test_hfl_negative_rounds_raises():
    model = TinyNet()
    with pytest.raises(ValueError, match="must be >= 0"):
        run_hfl(model, _loaders(), _edges(), _test_loader(),
                n_rounds=-1, clients_per_round=2, local_epochs=1)
