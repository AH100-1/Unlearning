import torch
import torch.nn as nn
import torchvision.models as models


_CIFAR10_NC = 10


def _patch_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    # CIFAR-10 (32x32): use 3x3 conv, stride=1, no maxpool
    if hasattr(model, "conv1"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if hasattr(model, "maxpool"):
        model.maxpool = nn.Identity()
    # replace FC head
    if hasattr(model, "fc"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    return model


def get_model(name: str, num_classes: int = _CIFAR10_NC, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name in ["resnet18", "r18"]:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name in ["resnet34", "r34"]:
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name in ["resnet50", "r50"]:
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return _patch_for_cifar(m, num_classes)