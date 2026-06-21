"""3D Dense Layer (B2).

Implements DenseLayer as described in Alalwan et al. (2021):
  Bottleneck: Conv3D 1x1x1 -> 128 channels -> BN -> ReLU
  Growth:     DS-Conv 3x3x3 -> 32 channels -> BN -> ReLU

Returns ONLY the 32-channel new features. Concatenation is the DenseBlock's job.

Conv -> BN -> ReLU ordering throughout.
"""

import torch
import torch.nn as nn

from dense_unet_3d.model.building_blocks.ds_conv import DepthwiseSeparableConv3d

BOTTLENECK_CHANNELS: int = 128
GROWTH_RATE: int = 32


class DenseLayer(nn.Module):
    """Single dense layer: 1x1x1 bottleneck then DS-Conv 3x3x3 growth step.

    Args:
        in_channels: Number of input channels (grows as the block accumulates).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels, BOTTLENECK_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm3d(BOTTLENECK_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.growth_stage = nn.Sequential(
            DepthwiseSeparableConv3d(
                in_channels=BOTTLENECK_CHANNELS,
                out_channels=GROWTH_RATE,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(GROWTH_RATE),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 32-channel new features only (spatial dims preserved)."""
        x = self.bottleneck(x)
        x = self.growth_stage(x)
        return x
