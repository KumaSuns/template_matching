"""
フレーム分類モデル。torchvision の ResNet をバックボーンに使用。
事前学習重みに合わせて ImageNet 正規化を推奨。
"""
import torch
import torch.nn as nn

# 事前学習 ResNet 用（学習・推論で同じにする）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

try:
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    resnet18 = None
    ResNet18_Weights = None


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    if resnet18 is None:
        raise ImportError("torchvision が必要です: pip install torchvision")
    if pretrained and ResNet18_Weights is not None:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
