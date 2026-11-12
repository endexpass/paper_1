"""Checkpoint and result persistence for Paper 1 FL experiments.

Design invariants:
- Atomic writes: write to <path>.tmp then rename — never a partially-written file.
- Tamper detection: sha256 hash over the payload; verified on load.
- Metadata preserved exactly (dict, not reordered).
- All torch saves use CPU tensors (aggregation is CPU-side).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch


def _hash_dict(payload: dict) -> str:
    """sha256 over canonical JSON of scalar fields + tensor hashes."""
    h = hashlib.sha256()
    for key in sorted(payload.keys()):
        val = payload[key]
        h.update(key.encode("utf-8"))
        if isinstance(val, dict):
            for k2 in sorted(val.keys()):
                v2 = val[k2]
                if isinstance(v2, torch.Tensor):
                    h.update(v2.detach().cpu().numpy().tobytes())
                else:
                    h.update(str(v2).encode("utf-8"))
        elif isinstance(val, torch.Tensor):
            h.update(val.detach().cpu().numpy().tobytes())
        else:
            h.update(json.dumps(val, sort_keys=True).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def save_checkpoint(
    state: dict,
    path: Path,
    optimizer_state: Optional[dict] = None,
    round_num: int = 0,
    metadata: Optional[dict] = None,
) -> str:
    """Save a model checkpoint atomically. Returns the payload hash."""
    payload = {
        "state": {k: v.detach().cpu().clone() for k, v in state.items()},
        "round": int(round_num),
        "metadata": metadata or {},
        "optimizer": optimizer_state if optimizer_state is not None else {},
    }
    payload["hash"] = _hash_dict(payload)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return payload["hash"]


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint, verify hash. Returns the full payload."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = payload.get("hash")
    # recompute without 'hash' field to verify
    body = {k: v for k, v in payload.items() if k != "hash"}
    actual = _hash_dict(body)
    if expected != actual:
        raise RuntimeError(
            f"checkpoint hash mismatch: expected {expected}, got {actual}. "
            f"File may be corrupt or tampered."
        )
    return payload


def save_result(result: dict, path: Path) -> None:
    """Save a JSON result dict atomically."""
    _atomic_write_text(path, json.dumps(result, indent=2, sort_keys=True))


def load_result(path: Path) -> dict:
    """Load a JSON result dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
