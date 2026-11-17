"""
Stage 4 - CNN patch-level classification model.

A convolutional neural network that classifies a hyperspectral patch into one of
the waterlogging classes:

  0 - waterlogged_soil
  1 - healthy_crop
  2 - stressed_crop
  3 - open_water
  4 - non_agricultural

Built with PyTorch. Supports custom CNN, ResNet-18, ResNet-50, and EfficientNet
backbones (the latter two via torchvision when available).
"""

from __future__ import annotations

import logging
from typing import Any

import config
import torch
import torch.nn as nn

LOGGER = logging.getLogger("deepwaterlog.cnn_model")


class CustomCNN(nn.Module):
    """
    A lightweight CNN for hyperspectral patch classification.

    Input : (batch, bands, patch_size, patch_size)
    Output: (batch, num_classes) logits
    """

    def __init__(
        self,
        in_channels: int | None = None,
        num_classes: int | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_channels = in_channels or config.CNN_INPUT_CHANNELS
        num_classes = num_classes or config.CNN_NUM_CLASSES

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Dropout2d(dropout * 0.5),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Dropout2d(dropout * 0.5),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
            nn.Dropout2d(dropout),
        )

        # Classifier head: adaptively pool to fixed size
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def build_cnn(
    architecture: str | None = None,
    in_channels: int | None = None,
    num_classes: int | None = None,
    pretrained: bool = False,
) -> nn.Module:
    """
    Build a CNN model.

    ``architecture``:
      - "custom_cnn"   : built-from-scratch CustomCNN
      - "resnet18"     : torchvision ResNet-18 (1-channel conv replaced)
      - "resnet50"     : torchvision ResNet-50 (1-channel conv replaced)
      - "efficientnet_b0": torchvision EfficientNet-B0 (first conv replaced)

    Returns an nn.Module producing raw logits.
    """
    architecture = (architecture or config.CNN_ARCHITECTURE).lower()
    in_channels = in_channels or config.CNN_INPUT_CHANNELS
    num_classes = num_classes or config.CNN_NUM_CLASSES

    if architecture == "custom_cnn":
        model = CustomCNN(in_channels=in_channels, num_classes=num_classes)
        LOGGER.info("Built CustomCNN (in_channels=%d, num_classes=%d).", in_channels, num_classes)
        return model

    try:
        import torchvision.models as models
    except ImportError as error:
        raise ImportError(
            f"Architecture '{architecture}' requires torchvision. Install it or "
            "use 'custom_cnn'."
        ) from error

    if architecture == "resnet18":
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "resnet50":
        model = models.resnet50(weights="DEFAULT" if pretrained else None)
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        # Replace first conv to accept in_channels
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown CNN architecture: {architecture}")

    LOGGER.info(
        "Built %s (in_channels=%d, num_classes=%d, pretrained=%s).",
        architecture, in_channels, num_classes, pretrained,
    )
    return model
