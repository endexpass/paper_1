"""MobileNetV2 backbone for Paper 1 FL experiments.

Design invariants (see CONTRACT.md §16):
- Same architecture for CIFAR-100 and Wheat.
- Only `num_classes` differs (100 vs 2).
- ImageNet-pretrained backbone by default.
- Standard MobileNetV2 BatchNorm behavior. FL training/aggregation
  policy is decided in flat_fl.py / hfl.py, not here.
- Input resolution: 224x224. CIFAR-100 images are resized from 32x32
  by data.py before reaching the model.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a MobileNetV2 with a fresh num_classes-head classifier.

    Args:
        num_classes: number of output logits (100 for CIFAR-100, 2 for Wheat).
        pretrained:  load ImageNet weights for the backbone (default True).

    Returns:
        A torchvision MobileNetV2 with classifier replaced by
        Dropout(0.2) -> Linear(in_features, num_classes).
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    return model
