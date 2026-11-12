"""Unit tests for fl_core.compression (E4 — uniform 8-bit)."""
import torch
import pytest

from fl_core.compression import (
    compress,
    decompress,
    dequantize_8bit,
    max_roundtrip_error,
    quantize_8bit,
)


def _state(shape=(8,), lo=-1.0, hi=1.0) -> dict:
    n = shape[0] if len(shape) == 1 else shape[0] * shape[1]
    t = torch.linspace(lo, hi, steps=n)
    return {"w": t.reshape(shape)}


def test_none_is_identity():
    s = _state()
    assert compress(s, "none") is s
    assert decompress(s, "none") is s


def test_uniform_8bit_returns_uint8_and_metadata():
    q, params = compress(_state(), "uniform_8bit")
    assert q["w"].dtype == torch.uint8
    assert set(params["w"].keys()) == {"scale", "min"}


def test_roundtrip_error_within_halfstep():
    s = _state((64,), lo=-2.5, hi=3.7)
    q, params = quantize_8bit(s)
    back = dequantize_8bit(q, params)
    err = (back["w"] - s["w"]).abs().max().item()
    rng = (s["w"].max() - s["w"].min()).item()
    assert err <= rng / 255.0 / 2 + 1e-6


def test_zero_range_exact_roundtrip():
    s = {"w": torch.full((4,), 3.14)}
    q, params = quantize_8bit(s)
    back = dequantize_8bit(q, params)
    assert torch.allclose(back["w"], s["w"], atol=1e-6)


def test_negative_values_preserved():
    s = {"w": torch.tensor([-5.0, -2.5, 0.0, 2.5, 5.0])}
    q, params = quantize_8bit(s)
    back = dequantize_8bit(q, params)
    assert back["w"].min().item() < 0
    assert back["w"].max().item() > 0


def test_compression_reduces_bytes():
    s = _state((100,))
    q, _ = quantize_8bit(s)
    assert q["w"].numel() * 1 < s["w"].numel() * 4


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown compression method"):
        compress(_state(), "fp4")


def test_deterministic():
    s = _state((32,), lo=-1.0, hi=2.0)
    q1, p1 = quantize_8bit(s)
    q2, p2 = quantize_8bit(s)
    assert torch.equal(q1["w"], q2["w"])
    assert torch.equal(p1["w"]["scale"], p2["w"]["scale"])
    assert torch.equal(p1["w"]["min"], p2["w"]["min"])


def test_max_roundtrip_error():
    assert max_roundtrip_error("none") == 0.0
    assert 0 < max_roundtrip_error("uniform_8bit") <= 1.0


def test_real_state_dict():
    import torch.nn as nn
    layer = nn.Linear(8, 3)
    s = layer.state_dict()
    q, params = quantize_8bit(s)
    back = dequantize_8bit(q, params)
    for k in s:
        assert back[k].shape == s[k].shape
        if s[k].numel() > 1:
            rng = (s[k].max() - s[k].min()).item()
            assert (back[k] - s[k]).abs().max().item() <= rng / 255.0 / 2 + 1e-5
