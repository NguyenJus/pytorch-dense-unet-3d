from typing import cast

import torch
from torch import nn


class TransitionBlock(nn.Module):
    """Transition layer between dense blocks (Alalwan et al. 2021, §3).

    Architecture (in order):
        BN -> Conv3d 1×1×1 (compression) -> Conv3d 1×1×1 stride (1,2,2) (spatial downsample)

    The compression step reduces channels from ``in_channels`` to
    ``round(in_channels * compression)`` — the **compression target**.

    With the default ``compression=0.5`` (paper §3) the compression target is
    ``round(in_channels * 0.5)``.  Small deviations from exactly 0.5 are
    possible when reconciling the overall model to ~3.6M parameters (see
    ``docs/research/2026-06-21-denseunet569-architecture-decisions.md``), but
    the default used here is the paper's stated value of 0.5.

    Downsampling is performed by the strided 1×1×1 convolution (stride
    ``(1, 2, 2)``), which halves H and W while leaving the depth dimension (D)
    unchanged.  This replaces the original pooling-based implementation.

    Args:
        in_channels: Number of input feature channels (output of the preceding
            dense block, i.e. ``initial_channels + count * growth_rate``).
        compression: Channel compression factor (default ``0.5`` per paper).
            Output channels = ``round(in_channels * compression)``.
    """

    def __init__(self, in_channels: int, compression: float = 0.5) -> None:
        super().__init__()
        out_channels: int = round(in_channels * compression)
        self.out_channels = out_channels
        self.convs = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False),
            nn.Conv3d(
                out_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transition block.

        Args:
            x: Input tensor of shape ``(N, in_channels, D, H, W)``.

        Returns:
            Output tensor of shape
            ``(N, round(in_channels * compression), D, H//2, W//2)``.
        """
        return cast(torch.Tensor, self.convs(x))
