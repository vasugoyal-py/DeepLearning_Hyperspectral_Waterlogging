"""Smoke tests for CNN and U-Net models (Stage 4)."""

import config
import torch

from core.cnn_model import build_cnn
from core.unet_model import build_unet


def test_cnn_forward_pass():
    model = build_cnn(architecture="custom_cnn")
    model.eval()
    batch = torch.randn(2, config.CNN_INPUT_CHANNELS, 64, 64)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2, config.CNN_NUM_CLASSES)


def test_unet_forward_pass():
    model = build_unet(architecture="unet")
    model.eval()
    batch = torch.randn(2, config.UNET_INPUT_CHANNELS, 64, 64)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2, config.UNET_NUM_CLASSES, 64, 64)
