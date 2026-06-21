"""3D Depthwise-Separable Convolution (B1).

Implements DepthwiseSeparableConv3d as described in Alalwan et al. (2021):
  - Depthwise stage: 3×3×3 conv with groups=in_channels (one filter per channel).
  - Pointwise stage: 1×1×1 conv mixing channels (in_channels → out_channels).

The kernel_size, padding, and stride on the depthwise stage are parameterizable;
the pointwise stage always uses kernel_size=1, stride=1, padding=0.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv3d(nn.Module):
    """3D depthwise-separable convolution: depthwise 3×3×3 + pointwise 1×1×1.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (pointwise stage).
        kernel_size: Kernel size for the depthwise stage. Default 3.
        stride: Stride for the depthwise stage. Default 1.
        padding: Padding for the depthwise stage. Default 1 (same-resolution for k=3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
