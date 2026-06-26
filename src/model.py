"""Model factory built on torchvision backbones.

Using torchvision keeps inference dependency-light: the grading platform only
needs ``torch`` and ``torchvision`` to rebuild and load the checkpoint, which
removes the whole class of "timm missing -> fall back -> predict everything as
one class" failures.
"""
from typing import Tuple

import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = model_name.lower()

    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    if name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


SUPPORTED_MODELS: Tuple[str, ...] = (
    "convnext_tiny",
    "convnext_small",
    "efficientnet_v2_s",
    "resnet50",
)
