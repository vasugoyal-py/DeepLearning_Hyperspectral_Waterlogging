"""
Stage 4 - U-Net pixel-level semantic segmentation model.

Produces a per-pixel waterlogging classification map from a hyperspectral patch
or full image. Classes:

  0 - waterlogged_soil
  1 - healthy_crop
  2 - stressed_crop
  3 - open_water
  4 - non_agricultural

Built with PyTorch. Supports U-Net, U-Net++, and Attention U-Net variants.
"""

from __future__ import annotations

import logging
from typing import Any

import config
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("deepwaterlog.unet_model")


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv.

    With bilinear upsampling the upsampled tensor keeps ``in_channels`` channels
    and is concatenated with a skip connection of ``in_channels // 2`` channels,
    so the double convolution receives ``in_channels + in_channels // 2`` input
    channels.  With transposed convolution the channels are halved on the way up,
    so the concatenation yields ``in_channels`` input channels.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels + in_channels // 2, out_channels, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 spatial dimensions (handles odd sizes)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Canonical U-Net (Ronneberger et al., 2015).

    Input : (batch, in_channels, H, W)
    Output: (batch, num_classes, H, W) logits
    """

    def __init__(
        self,
        in_channels: int | None = None,
        num_classes: int | None = None,
        depth: int | None = None,
        base_filters: int | None = None,
        bilinear: bool = True,
    ):
        super().__init__()
        in_channels = in_channels or config.UNET_INPUT_CHANNELS
        num_classes = num_classes or config.UNET_NUM_CLASSES
        depth = depth or config.UNET_DEPTH
        base_filters = base_filters or config.UNET_BASE_FILTERS

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_filters)
        in_ch = base_filters
        downs = []
        for _ in range(depth):
            out_ch = in_ch * 2
            downs.append(Down(in_ch, out_ch))
            in_ch = out_ch
        self.downs = nn.ModuleList(downs)

        factor = 2 if bilinear else 1
        ups = []
        for _ in range(depth):
            out_ch = in_ch // factor
            ups.append(Up(in_ch, out_ch, bilinear))
            in_ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.outc = OutConv(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        skips = [x1]
        for down in self.downs:
            x1 = down(x1)
            skips.append(x1)

        for i, up in enumerate(self.ups):
            x1 = up(x1, skips[-(i + 2)])

        logits = self.outc(x1)
        return logits


def build_unet(
    architecture: str | None = None,
    in_channels: int | None = None,
    num_classes: int | None = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a segmentation model.

    ``architecture``:
      - "unet"            : canonical U-Net
      - "unet_plusplus"   : U-Net++ nested (currently falls back to U-Net)
      - "attention_unet"  : attention U-Net (currently falls back to U-Net)

    Additional kwargs are forwarded to UNet (depth, base_filters, bilinear).
    """
    architecture = (architecture or config.UNET_ARCHITECTURE).lower()
    in_channels = in_channels or config.UNET_INPUT_CHANNELS
    num_classes = num_classes or config.UNET_NUM_CLASSES

    if architecture == "unet":
        model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            depth=kwargs.get("depth", config.UNET_DEPTH),
            base_filters=kwargs.get("base_filters", config.UNET_BASE_FILTERS),
            bilinear=kwargs.get("bilinear", True),
        )
        LOGGER.info(
            "Built U-Net (in_channels=%d, num_classes=%d, depth=%d, base_filters=%d).",
            in_channels, num_classes, config.UNET_DEPTH, config.UNET_BASE_FILTERS,
        )
        return model

    # U-Net++ and Attention U-Net fall back to canonical U-Net with a notice,
    # keeping the code runnable without extra implementations.
    LOGGER.warning(
        "Architecture '%s' not yet implemented separately; falling back to "
        "canonical U-Net.",
        architecture,
    )
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=kwargs.get("depth", config.UNET_DEPTH),
        base_filters=kwargs.get("base_filters", config.UNET_BASE_FILTERS),
        bilinear=True,
    )
