"""Explicit diagnostic Fig. 1 reconstruction; unresolved choices are not paper facts.

The historical model remains in DenseUNet3d.py. See the topology evidence record.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


def batch_norm(channels: int) -> nn.BatchNorm3d:
    return nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)


class SamePad3d(nn.Module):
    """TensorFlow SAME_UPPER geometry, including asymmetric stride padding."""

    def __init__(self, kernel: int, stride: int, value: float = 0.0) -> None:
        super().__init__()
        self.kernel, self.stride, self.value = kernel, stride, value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pads: list[int] = []
        for size in reversed(x.shape[-3:]):
            total = max((math.ceil(size / self.stride) - 1) * self.stride + self.kernel - size, 0)
            pads.extend((total // 2, total - total // 2))
        return F.pad(x, pads, value=self.value)


class FigureDenseLayer(nn.Module):
    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv3d(channels, bottleneck, 1), batch_norm(bottleneck), nn.ReLU()
        )
        self.growth = nn.Sequential(
            nn.Conv3d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck),
            nn.Conv3d(bottleneck, 32, 1),
            batch_norm(32),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.growth(self.bottleneck(x)))


class FigureDenseBlock(nn.Module):
    def __init__(self, channels: int, count: int, bottleneck: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            FigureDenseLayer(channels + i * 32, bottleneck) for i in range(count)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.cat((x, layer(x)), dim=1)
        return x


class FigureUpsample(nn.Module):
    def __init__(
        self, channels: int, skip_channels: int, out: int, target: tuple[int, int, int]
    ) -> None:
        super().__init__()
        self.target = target
        self.conv = nn.Sequential(
            nn.Conv3d(channels + skip_channels, out, 3, padding=1), batch_norm(out), nn.ReLU()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Dashed figure sources are below target resolution; both paths resize.
        x = F.interpolate(x, size=self.target, mode="trilinear", align_corners=False)
        skip = F.interpolate(skip, size=self.target, mode="trilinear", align_corners=False)
        return cast(torch.Tensor, self.conv(torch.cat((x, skip), dim=1)))


class FigureSkipReconstruction(nn.Module):
    """One bounded candidate, with fixed 224×224×12 input and diagnostic status."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            SamePad3d(7, 2), nn.Conv3d(1, 96, 7, stride=2), batch_norm(96), nn.ReLU()
        )
        self.pool = nn.Sequential(SamePad3d(3, 2, -math.inf), nn.MaxPool3d(3, 2))
        c = 96
        channels = []
        for i, (count, bottleneck) in enumerate(
            zip((4, 12, 24, 36), (128, 128, 128, 32), strict=True), 1
        ):
            setattr(self, f"dense_block{i}", FigureDenseBlock(c, count, bottleneck))
            c += 32 * count
            channels.append(c)
            if i < 4:
                out = c // 2
                setattr(
                    self,
                    f"transition{i}",
                    nn.Sequential(
                        nn.Conv3d(c, out, 1),
                        batch_norm(out),
                        nn.ReLU(),
                        nn.Conv3d(out, out, 1, stride=(1, 2, 2)),
                    ),
                )
                c = out
        self.skip_sources = ("dense_block4", "dense_block3", "dense_block2", "dense_block1", "stem")
        self.encoder_channels = tuple(channels)
        targets = ((3, 14, 14), (3, 28, 28), (3, 56, 56), (6, 112, 112), (12, 224, 224))
        for i, (skip, out, target) in enumerate(
            zip((*reversed(channels), 96), (504, 224, 192, 96, 64), targets, strict=True), 1
        ):
            setattr(self, f"up{i}", FigureUpsample(c, skip, out, target))
            c = out
        self.classifier = nn.Conv3d(64, 3, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            if module.groups == module.in_channels and module.groups > 1:
                # Keras-style depthwise kernel layout (kD,kH,kW,C,multiplier).
                volume = math.prod(module.kernel_size)
                fan_sum = volume * (module.in_channels + module.out_channels // module.in_channels)
                nn.init.uniform_(module.weight, -math.sqrt(6 / fan_sum), math.sqrt(6 / fan_sum))
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or tuple(x.shape[1:]) != (1, 12, 224, 224):
            raise ValueError(
                f"Reconstruction requires NCDHW input (N,1,12,224,224); got {tuple(x.shape)}"
            )
        stem = self.stem(x)
        x = self.pool(stem)
        dense = []
        for i in range(1, 5):
            x = getattr(self, f"dense_block{i}")(x)
            dense.append(x)
            if i < 4:
                x = getattr(self, f"transition{i}")(x)
        for i, skip in enumerate((*reversed(dense), stem), 1):
            x = getattr(self, f"up{i}")(x, skip)
        return cast(torch.Tensor, self.classifier(x))
