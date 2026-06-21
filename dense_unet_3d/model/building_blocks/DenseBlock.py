"""Dense Block with REAL dense connectivity (B3).

Each DenseLayer consumes the running torch.cat of the block input + all prior
layer outputs.  The block returns the full concatenation, so output channels =
in_channels + count * growth.

This fixes the original nn.Sequential-no-cat bug.
"""

import torch
import torch.nn as nn

from dense_unet_3d.model.building_blocks.dense_layer import DenseLayer


class DenseBlock(nn.Module):
    """Stack of DenseLayers with real dense (DenseNet) connectivity.

    Args:
        in_channels: Number of channels entering the block.
        growth: Growth rate — channels added per layer. Default 32.
        count: Number of DenseLayers in the block.
    """

    def __init__(
        self,
        in_channels: int,
        count: int,
        growth: int = 32,
    ) -> None:
        super().__init__()
        self.growth = growth
        # Build each layer; layer i receives in_channels + i*growth channels.
        layers: list[DenseLayer] = []
        running_channels = in_channels
        for _ in range(count):
            layers.append(DenseLayer(in_channels=running_channels))
            running_channels += growth
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with running concatenation across all layers.

        Args:
            x: Input tensor of shape (N, in_channels, D, H, W).

        Returns:
            Tensor of shape (N, in_channels + count*growth, D, H, W).
        """
        features = x
        for layer in self.layers:
            new_feat = layer(features)
            features = torch.cat([features, new_feat], dim=1)
        return features
