"""Unit tests for fl_core.aggregation (Stage 4a)."""
import pytest
import torch

from fl_core.aggregation import (
    aggregate,
    fedavg,
    fedavg_simple,
    scaffold_aggregate,
)


def _state(value: float, shape=(4,)) -> dict:
    return {"w": torch.full(shape, value)}


def test_fedavg_equal_weights_is_mean():
    out = fedavg([_state(1.0), _state(3.0)], [1, 1])
    assert torch.allclose(out["w"], torch.full((4,), 2.0))


def test_fedavg_weighted():
    out = fedavg([_state(1.0), _state(3.0)], [1, 3])  # (1*1 + 3*3)/4 = 2.5
    assert torch.allclose(out["w"], torch.full((4,), 2.5))


def test_fedavg_single_client_identity():
    a = _state(7.5)
    out = fedavg([a], [100])
    assert torch.allclose(out["w"], a["w"].float())


def test_fedavg_zero_weight_ignored():
    out = fedavg([_state(1.0), _state(999.0)], [1, 0])
    assert torch.allclose(out["w"], torch.full((4,), 1.0))


def test_fedavg_all_zero_weights_raises():
    with pytest.raises(ValueError, match="sum of client_weights is 0"):
        fedavg([_state(1.0)], [0])


def test_fedavg_negative_weight_raises():
    with pytest.raises(ValueError, match="non-negative"):
        fedavg([_state(1.0)], [-1])


def test_fedavg_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        fedavg([_state(1.0)], [1, 2])


def test_fedavg_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        fedavg([_state(1.0, shape=(4,)), _state(3.0, shape=(5,))], [1, 1])


def test_fedavg_key_mismatch_raises():
    a = {"w": torch.ones(4)}
    b = {"other": torch.ones(4)}
    with pytest.raises(ValueError, match="different keys"):
        fedavg([a, b], [1, 1])


def test_fedavg_output_on_cpu():
    out = fedavg([_state(1.0)], [1])
    for v in out.values():
        assert v.device.type == "cpu"


def test_fedavg_simple_uniform():
    out = fedavg_simple([_state(0.0), _state(2.0), _state(4.0)])
    assert torch.allclose(out["w"], torch.full((4,), 2.0))


def test_fedavg_commutative():
    a, b = _state(1.0), _state(3.0)
    out1 = fedavg([a, b], [1, 2])
    out2 = fedavg([b, a], [2, 1])
    assert torch.allclose(out1["w"], out2["w"])


def test_scaffold_aggregate_returns_model_and_control():
    states = [_state(1.0), _state(3.0)]
    deltas = [_state(0.1), _state(0.2)]
    control = {"w": torch.zeros(4)}
    new_model, new_control = scaffold_aggregate(states, deltas, [1, 1], control, server_lr=1.0)
    assert torch.allclose(new_model["w"], torch.full((4,), 2.0))
    assert torch.allclose(new_control["w"], torch.full((4,), 0.15))


def test_aggregate_dispatches_fedavg():
    out = aggregate("fedavg", [_state(1.0), _state(3.0)], [1, 1])
    assert torch.allclose(out["w"], torch.full((4,), 2.0))


def test_aggregate_fedprox_same_as_fedavg():
    out = aggregate("fedprox", [_state(1.0), _state(3.0)], [1, 1])
    assert torch.allclose(out["w"], torch.full((4,), 2.0))


def test_aggregate_scaffold_requires_kwargs():
    with pytest.raises(ValueError, match="scaffold requires"):
        aggregate("scaffold", [_state(1.0)], [1])


def test_aggregate_unknown_raises():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate("ditto", [_state(1.0)], [1])
