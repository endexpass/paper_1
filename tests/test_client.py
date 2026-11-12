"""Unit tests for fl_core.client (Stage 3c)."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_core.client import evaluate_local, train_local


class TinyNet(nn.Module):
    def __init__(self, n_classes: int = 3, input_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def _make_data(n: int = 64, n_classes: int = 3, input_dim: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, input_dim, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return TensorDataset(x, y)


def _loader(ds, batch_size: int = 16):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def test_train_local_reduces_loss():
    torch.manual_seed(0)
    ds = _make_data(n=128, seed=0)
    loader = _loader(ds)

    model = TinyNet()
    initial = evaluate_local(model, loader, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

    train_local(model, loader, local_epochs=5, optimizer_name="adam", lr=0.05)

    final = evaluate_local(model, loader, loss_fn=nn.CrossEntropyLoss(reduction="sum"))
    assert final["loss"] < initial["loss"]


def test_train_local_returns_cpu_state():
    model = TinyNet()
    loader = _loader(_make_data(n=16))
    state = train_local(model, loader, local_epochs=1)
    for k, v in state.items():
        assert isinstance(v, torch.Tensor)
        assert v.device.type == "cpu"


def test_train_local_sgd():
    model = TinyNet()
    loader = _loader(_make_data(n=32))
    state = train_local(model, loader, local_epochs=1, optimizer_name="sgd", lr=0.05)
    assert isinstance(state, dict)
    assert "fc.weight" in state


def test_train_local_adam():
    model = TinyNet()
    loader = _loader(_make_data(n=32))
    state = train_local(model, loader, local_epochs=1, optimizer_name="adam", lr=0.05)
    assert isinstance(state, dict)


def test_train_local_zero_epochs_returns_unchanged():
    torch.manual_seed(0)
    model = TinyNet()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = _loader(_make_data(n=32))
    state = train_local(model, loader, local_epochs=0)
    for k in before:
        assert torch.equal(before[k], state[k])


def test_train_local_negative_epochs_raises():
    model = TinyNet()
    loader = _loader(_make_data(n=16))
    with pytest.raises(ValueError, match="must be >= 0"):
        train_local(model, loader, local_epochs=-1)


def test_train_local_unknown_optimizer_raises():
    model = TinyNet()
    loader = _loader(_make_data(n=16))
    with pytest.raises(ValueError, match="Unknown optimizer"):
        train_local(model, loader, local_epochs=1, optimizer_name="rmsprop")


def test_evaluate_local_returns_metrics():
    model = TinyNet()
    loader = _loader(_make_data(n=32))
    metrics = evaluate_local(model, loader)
    assert set(metrics.keys()) == {"loss", "accuracy", "n_samples"}
    assert metrics["n_samples"] == 32
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] >= 0.0


def test_evaluate_local_perfect_accuracy_on_trivial_data():
    torch.manual_seed(0)
    x = torch.randn(64, 1, 8)
    y = (x[:, 0, 0] > 0).long()
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = TinyNet(n_classes=2, input_dim=8)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        model.fc.weight[1, 0] = 1.0

    metrics = evaluate_local(model, loader)
    assert metrics["accuracy"] > 0.95


def test_evaluate_local_empty_returns_zero():
    x = torch.empty(0, 1, 8)
    y = torch.empty(0, dtype=torch.long)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    model = TinyNet()
    metrics = evaluate_local(model, loader)
    assert metrics["n_samples"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["loss"] == 0.0
