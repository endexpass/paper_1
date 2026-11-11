"""Unit tests for fl_core.model (Stage 1)."""
import pytest
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from fl_core.model import build_model


def test_build_model_returns_mobilenetv2():
    model = build_model(num_classes=10, pretrained=False)
    assert type(model).__name__ == "MobileNetV2"


def test_classifier_is_dropout_plus_linear():
    model = build_model(num_classes=42, pretrained=False)
    clf = model.classifier
    assert isinstance(clf, nn.Sequential)
    assert len(clf) == 2
    assert isinstance(clf[0], nn.Dropout)
    assert clf[0].p == 0.2
    assert isinstance(clf[1], nn.Linear)


def test_classifier_output_dim_matches_num_classes():
    for n in (2, 10, 100):
        model = build_model(num_classes=n, pretrained=False)
        linear = model.classifier[1]
        assert linear.out_features == n
        assert linear.in_features == 1280


@pytest.mark.parametrize("num_classes", [2, 100])
def test_forward_shape(num_classes):
    model = build_model(num_classes=num_classes, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, num_classes)
    assert torch.isfinite(y).all()


def test_pretrained_backbone_loads():
    model = build_model(num_classes=100, pretrained=True)
    clf = model.classifier[1]
    assert isinstance(clf, nn.Linear)
    assert clf.out_features == 100


def test_pretrained_false_does_not_touch_cache(monkeypatch):
    called = {"n": 0}

    def fake_mobilenet_v2(weights=None, **kw):
        called["n"] += 1
        assert weights is None
        return mobilenet_v2(weights=None)

    import fl_core.model as m
    monkeypatch.setattr(m, "mobilenet_v2", fake_mobilenet_v2)
    build_model(num_classes=2, pretrained=False)
    assert called["n"] == 1


def test_invalid_num_classes_raises():
    with pytest.raises(ValueError, match="num_classes must be >= 1"):
        build_model(num_classes=0, pretrained=False)
