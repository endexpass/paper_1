"""Model compression for Paper 1 FL experiments (E4 — uniform 8-bit).

Contract (CONTRACT.md §14, E4):
  quantize:   float32 tensor → uint8 tensor + {scale, min}
  dequantize: uint8 tensor + {scale, min} → float32 tensor
  invariant:  dequantize(quantize(x)) ≈ x within half-step tolerance

No speculative extensions. Zero-range tensors round-trip exactly.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

QuantPayload = Tuple[Dict[str, torch.Tensor], Dict[str, dict]]


def quantize_8bit(state: Dict[str, torch.Tensor]) -> QuantPayload:
    """Quantize each tensor to uint8 via per-tensor min/max scaling.

    Returns (quantized, params):
        quantized[k]: uint8 tensor, same shape as state[k]
        params[k]:    {"scale": scalar, "min": scalar}
    Reconstruction: state[k] ≈ quantized[k].float() * scale + min
    """
    quantized: Dict[str, torch.Tensor] = {}
    params: Dict[str, dict] = {}
    for k, v in state.items():
        flat = v.detach().cpu().float().flatten()
        if flat.numel() == 0:
            quantized[k] = torch.zeros_like(v, dtype=torch.uint8)
            params[k] = {"scale": torch.tensor(1.0), "min": torch.tensor(0.0)}
            continue
        mn = flat.min()
        mx = flat.max()
        rng = mx - mn
        if float(rng) == 0.0:
            quantized[k] = torch.zeros_like(v, dtype=torch.uint8)
            params[k] = {"scale": torch.tensor(1.0), "min": mn.to(torch.float32)}
        else:
            scale = rng / 255.0
            q = ((v.detach().cpu().float() - mn) / scale).round().clamp(0, 255).to(torch.uint8)
            quantized[k] = q
            params[k] = {"scale": scale.to(torch.float32), "min": mn.to(torch.float32)}
    return quantized, params


def dequantize_8bit(quantized: Dict[str, torch.Tensor], params: Dict[str, dict]) -> Dict[str, torch.Tensor]:
    """Reconstruct float32 tensors from uint8 + per-tensor scale/min."""
    out: Dict[str, torch.Tensor] = {}
    for k, q in quantized.items():
        p = params[k]
        out[k] = q.float() * p["scale"] + p["min"]
    return out


def compress(state: Dict[str, torch.Tensor], method: str = "none"):
    if method == "none":
        return state
    if method == "uniform_8bit":
        return quantize_8bit(state)
    raise ValueError(f"Unknown compression method: {method!r}. Expected 'none' or 'uniform_8bit'.")


def decompress(payload, method: str = "none") -> Dict[str, torch.Tensor]:
    if method == "none":
        return payload
    if method == "uniform_8bit":
        quantized, params = payload
        return dequantize_8bit(quantized, params)
    raise ValueError(f"Unknown compression method: {method!r}. Expected 'none' or 'uniform_8bit'.")


def max_roundtrip_error(method: str = "uniform_8bit") -> float:
    if method == "none":
        return 0.0
    if method == "uniform_8bit":
        return 0.5 / 255.0
    raise ValueError(f"Unknown compression method: {method!r}")
