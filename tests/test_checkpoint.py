"""Unit tests for fl_core.checkpoint (Stage 5a)."""
import json
from pathlib import Path

import pytest
import torch

from fl_core.checkpoint import (
    _hash_dict,
    load_checkpoint,
    load_result,
    save_checkpoint,
    save_result,
)


def _state() -> dict:
    return {
        "w": torch.randn(4, 4),
        "b": torch.zeros(4),
    }


def test_save_load_roundtrip_state(tmp_path):
    state = _state()
    h = save_checkpoint(state, tmp_path / "ckpt.pt", round_num=5, metadata={"exp_id": "E0"})
    loaded = load_checkpoint(tmp_path / "ckpt.pt")
    for k in state:
        assert torch.allclose(loaded["state"][k], state[k])
    assert loaded["round"] == 5
    assert loaded["metadata"] == {"exp_id": "E0"}
    assert loaded["hash"] == h


def test_save_load_with_optimizer(tmp_path):
    state = _state()
    opt = {"momentum": {k: torch.zeros_like(v) for k, v in state.items()}}
    save_checkpoint(state, tmp_path / "ckpt.pt", optimizer_state=opt, round_num=3)
    loaded = load_checkpoint(tmp_path / "ckpt.pt")
    for k in opt["momentum"]:
        assert torch.equal(loaded["optimizer"]["momentum"][k], opt["momentum"][k])


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_hash_verification_detects_tamper(tmp_path):
    state = _state()
    save_checkpoint(state, tmp_path / "ckpt.pt")
    payload = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    payload["state"]["w"] = payload["state"]["w"] + 1.0
    torch.save(payload, tmp_path / "ckpt.pt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_checkpoint(tmp_path / "ckpt.pt")


def test_atomic_write(tmp_path):
    state = _state()
    save_checkpoint(state, tmp_path / "nested" / "deep" / "ckpt.pt")
    assert (tmp_path / "nested" / "deep" / "ckpt.pt").exists()
    assert not (tmp_path / "nested" / "deep" / "ckpt.pt.tmp").exists()


def test_result_roundtrip(tmp_path):
    result = {"loss": 0.42, "accuracy": 0.87, "n_samples": 1000}
    save_result(result, tmp_path / "result.json")
    loaded = load_result(tmp_path / "result.json")
    assert loaded == result


def test_result_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="result file not found"):
        load_result(tmp_path / "missing.json")


def test_result_atomic(tmp_path):
    save_result({"a": 1}, tmp_path / "deep" / "nested" / "result.json")
    assert (tmp_path / "deep" / "nested" / "result.json").exists()


def test_hash_deterministic():
    state = {"w": torch.ones(3)}
    h1 = _hash_dict({"state": state})
    h2 = _hash_dict({"state": state})
    assert h1 == h2


def test_hash_different_for_different_data():
    h1 = _hash_dict({"state": {"w": torch.ones(3)}})
    h2 = _hash_dict({"state": {"w": torch.zeros(3)}})
    assert h1 != h2
