"""Upsampling block for the 3D-DenseUNet-569 decoder (Task C2).

Uses a 3D depthwise-separable convolution (DS-Conv) in place of a dense
3×3×3 Conv3d, consistent with the paper's DS-Conv-throughout design.  This
reduces the decoder's parameter count from ~40M (dense 3×3×3) to ~2M,
which is necessary to land the full model within the 3.6M ±15% budget.
"""

from typing import cast

import torch
from torch import nn

from dense_unet_3d.model.building_blocks.ds_conv import DepthwiseSeparableConv3d


class UpsamplingBlock(nn.Module):
    """Upsampling block for the decoder path of 3D-DenseUNet-569.

    Upsamples the MAIN path only to target_size via trilinear interpolation,
    then concatenates the skip connection (already at target_size, NOT upsampled),
    then applies DS-Conv 3×3×3 → BN → ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        target_size: tuple[int, int, int],
    ) -> None:
        """Create the layers for the upsampling block.

        :param in_channels:   channels of the main (upsampled) input tensor.
        :param skip_channels: channels of the skip-connection tensor (already at target_size).
        :param out_channels:  number of output channels from this block.
        :param target_size:   (D, H, W) spatial size that main path is upsampled to.
        """
        super().__init__()
        self.upsample = nn.Upsample(size=target_size, mode="trilinear", align_corners=True)
        # DS-Conv (depthwise 3×3×3 + pointwise 1×1×1) with BN + ReLU.
        combined_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            DepthwiseSeparableConv3d(
                in_channels=combined_in,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x:    main path tensor — will be upsampled to target_size.
        :param skip: skip-connection tensor — already at target_size, NOT upsampled.
        :return:     output tensor with out_channels channels at target_size spatial dims.
        """
        x_up = self.upsample(x)
        combined = torch.cat((x_up, skip), dim=1)
        return cast(torch.Tensor, self.conv(combined))
