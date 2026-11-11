# COMMON/fl_core/model.py
"""
Model creation for Paper 1 experiments.

Implements CONTRACT.md Section 2 and frozen Decision 1:
  - Immutable ImageNet-initialized MobileNetV2 backbone cache.
  - Fresh model instance per call; no mutable instance is shared across
    architectures or runs.
  - Deterministic classifier head initialization from `init_seed`.
  - initial_state_hash() proves both architectures in a paired run started
    from identical weights (Preamble rule 3, R2).

No module-level dataset constants appear in this file (Preamble rule 2).
`num_classes` always flows through arguments.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# ---------------------------------------------------------------------------
# Immutable backbone cache (frozen Decision 1)
# ---------------------------------------------------------------------------
_BACKBONE_CACHE: Optional[dict[str, torch.Tensor]] = None
_CACHE_LOCK = threading.Lock()


def _get_backbone_state() -> dict[str, torch.Tensor]:
    """Return the cached ImageNet MobileNetV2 state dict.

    The cache is populated once per process and never mutated. Callers
    MUST NOT modify the returned tensors. `create_model` and `clone_model`
    copy tensor contents out of the cache; they never write back.

    The classifier head tensors are included in the cache (they belong to
    the 1000-class template). Callers must filter them out before loading
    into a model with a different `num_classes`.
    """
    global _BACKBONE_CACHE
    if _BACKBONE_CACHE is None:
        with _CACHE_LOCK:
            if _BACKBONE_CACHE is None:
                template = mobilenet_v2(weights="IMAGENET1K_V1")
                _BACKBONE_CACHE = {
                    k: v.detach().clone()
                    for k, v in template.state_dict().items()
                }
                del template
    return _BACKBONE_CACHE


def reset_cache() -> None:
    """Clear the backbone cache.

    Tests call this between cases so a cached template from one test cannot
    leak into the next. Production code never calls this.
    """
    global _BACKBONE_CACHE
    with _CACHE_LOCK:
        _BACKBONE_CACHE = None


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------
def _build_empty(num_classes: int) -> nn.Module:
    """MobileNetV2 with an `num_classes`-sized classifier, on CPU, uninitialized."""
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _init_head_cpu(model: nn.Module, init_seed: int) -> None:
    """Deterministically initialize `model.classifier[1]` on CPU.

    Uses a CPU generator so the head is identical regardless of the device
    the model is later moved to. This is what makes `initial_state_hash`
    comparable across CPU and CUDA runs of the same seed.

    Distribution is uniform(-1/sqrt(in_features), 1/sqrt(in_features)),
    the default Kaiming-uniform bound for Linear. The specific distribution
    does not matter scientifically as long as it is deterministic and
    identical for both architectures; the hash check enforces that.
    """
    linear = model.classifier[1]
    gen = torch.Generator(device="cpu").manual_seed(init_seed)
    bound = 1.0 / (linear.in_features ** 0.5)
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=gen)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound, generator=gen)


def _load_backbone(model: nn.Module, num_classes: int) -> None:
    """Load the cached ImageNet backbone into `model`, skipping classifier tensors.

    The cache holds a 1000-class classifier. The target model has `num_classes`
    classes. A non-strict `load_state_dict` on mismatched classifier shapes
    would raise; filtering the classifier keys out first is the clean fix.
    """
    state = _get_backbone_state()
    filtered = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    model.load_state_dict(filtered, strict=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_model(num_classes: int, init_seed: int, device: torch.device) -> nn.Module:
    """Create a fresh MobileNetV2 for `num_classes` classes.

    Steps (CONTRACT.md Section 2.1):
      1. Build an empty MobileNetV2 with a fresh `num_classes` classifier.
      2. Load the immutable ImageNet backbone (classifier tensors skipped).
      3. Re-initialize the classifier head deterministically from `init_seed`.
      4. Move to `device`.

    Two calls with identical `(num_classes, init_seed)` produce models whose
    `state_dict()` tensors are identical, on the same environment. This is
    what `initial_state_hash` verifies and what R2 guarantees.
    """
    model = _build_empty(num_classes)
    _load_backbone(model, num_classes)
    _init_head_cpu(model, init_seed)
    return model.to(device)


def clone_model(source: nn.Module, num_classes: int, device: torch.device) -> nn.Module:
    """Create an independent, deep-copied model with `num_classes` classifier.

    Used by `client.train_local` to give each client its own model.
    State-dict tensors are cloned; mutating the clone does not affect `source`.
    """
    clone = _build_empty(num_classes)
    clone.load_state_dict(
        {k: v.detach().clone() for k, v in source.state_dict().items()}
    )
    return clone.to(device)


def initial_state_hash(model: nn.Module) -> str:
    """Deterministic sha256 over `model.state_dict()`, sorted by key.

    Recorded in both Flat and HFL result JSONs (CONTRACT.md Section 2.3,
    Preamble rule 3, R2). A paired run whose two JSONs have different
    `initial_state_hash` values is invalid and must not enter the paper.

    Includes key name, tensor shape, dtype, and raw bytes so that two models
    differing only in shape or dtype are distinguished.
    """
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state.keys()):
        tensor = state[key]
        h.update(key.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()