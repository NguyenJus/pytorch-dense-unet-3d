"""Full 3D-DenseUNet-569 assembly (Task C3).

Faithful re-implementation of the encoder/decoder from Alalwan et al. (2021),
"Efficient 3D Deep Learning Model for Medical Image Semantic Segmentation,"
*Alexandria Engineering Journal* 60 (2021) 1231-1239. See the spec
``docs/superpowers/specs/2026-06-21-faithful-3d-denseunet569-design.md`` §2/§3
and the committed decision record
``docs/research/2026-06-21-denseunet569-architecture-decisions.md``.

Architecture (NCDHW; spatial shown as D x H x W):

  Encoder
    input          (N,    1, 12, 224, 224)
    stem  Conv3D k=7 s=2 pad=3 -> 96      (6, 112, 112)
    pool  MaxPool3D k=2 s=2     -> 96      (3,  56,  56)
    DB1   2 layers              -> 160     (3,  56,  56)
    T1    compress 0.5, stride  ->  80     (3,  28,  28)
    DB2   6 layers              -> 272     (3,  28,  28)
    T2    compress 0.5, stride  -> 136     (3,  14,  14)
    DB3   12 layers             -> 520     (3,  14,  14)
    T3    compress 0.5, stride  -> 260     (3,   7,   7)
    DB4   18 layers             -> 836     (3,   7,   7)

  Decoder (5 trilinear-upsample DS-Conv blocks, widths 504/224/192/96/64; U-Net skips)
    up1  -> 504   (3,  14,  14)  skip = DB3 ( 520 ch @ 14x14x3)
    up2  -> 224   (3,  28,  28)  skip = DB2 ( 272 ch @ 28x28x3)
    up3  -> 192   (3,  56,  56)  skip = DB1 ( 160 ch @ 56x56x3)
    up4  ->  96   (6, 112, 112)  skip = stem (  96 ch @ 112x112x6)
    up5  ->  64   (12, 224, 224) NO encoder skip (no feature exists at 224x224x12)
    classifier Conv3D 1x1x1 -> 3 (12, 224, 224)

Design decisions (the §3 "knobs"), recorded for the G1 decision record:

* Paddings (chosen to PRODUCE the §2 spatial dims; Fig 1 pad=0 labels ignored):
    - stem: Conv3D k=7 s=2 pad=3   -> 224->112, 12->6
    - pool: MaxPool3D k=2 s=2      -> 112->56, 6->3
    - dense layers / decoder convs: same-resolution k=3 pad=1; transition convs
      are k=1 (pad=0); transitions downsample H,W only via stride (1,2,2).
* Depth axis: halved at stem (12->6) and pool (6->3), then held at 3 through all
  dense blocks and transitions. The two highest-resolution decoder blocks (up4,
  up5) restore depth 3->6->12 via the trilinear Upsample ``size=`` target.
* Transition compression target: **0.5** (the paper's stated value, unchanged).
  This lands the model comfortably inside the 3.06M-4.14M band, so no deviation
  from 0.5 was required.
* Skip wiring (5 encoder levels feed the decoder): the decoder bottom is DB4 at
  7x7x3; each up-block takes its skip from the encoder level at the matching
  post-upsample resolution (DB3@14, DB2@28, DB1@56, stem@112). The 224x224x12
  level has no encoder feature, so up5 takes no skip (skip_channels=0).
* Block counts: paper specifies (4,12,24,36) at 1:3:6:9 ratio.  Full-scale
  yields ~10.8M params (outside 3.6M ±15% band) because the growing bottleneck
  1×1×1 in each DenseLayer takes the full concatenated input (in_ch → 128).
  Using half-scale (2,6,12,18) preserves the ratio and achieves ~3.52M within
  [3.06M, 4.14M].  Decision recorded in
  ``docs/research/2026-06-21-denseunet569-architecture-decisions.md``.
* Decoder convolutions: DS-Conv (depthwise 3×3×3 + pointwise 1×1×1) replaces
  the dense 3×3×3 Conv3d, consistent with the paper's DS-Conv-throughout design.
* Achieved trainable parameter count: ~3.52M, within the 3.6M +/- 15% band
  (3,060,000 - 4,140,000). The exact value is asserted (and printed) by
  ``tests/model/test_dense_unet_3d.py::test_param_count_within_band``.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from dense_unet_3d.model.building_blocks.DenseBlock import DenseBlock
from dense_unet_3d.model.building_blocks.TransitionBlock import TransitionBlock
from dense_unet_3d.model.building_blocks.UpsamplingBlock import UpsamplingBlock

# Encoder hyper-parameters (paper §2/§3).
_STEM_CHANNELS: int = 96
_GROWTH: int = 32
# AUTHORIZED DEVIATION: paper specifies (4, 12, 24, 36) at 1:3:6:9 ratio, but
# with real dense connectivity (bottleneck 1×1×1: in_ch → 128) that yields
# ~10.8M params — far outside the 3.6M ±15% target band.  These are HALF the
# paper's counts (2, 6, 12, 18), preserving the 1:3:6:9 ratio and landing at
# ~3.52M in [3.06M, 4.14M].  growth=32 and bottleneck=128 are UNCHANGED.
# Decision record: docs/research/2026-06-21-denseunet569-architecture-decisions.md
_BLOCK_COUNTS: tuple[int, int, int, int] = (2, 6, 12, 18)
_COMPRESSION: float = 0.5

# Decoder out-channel widths, layers 1..5 (paper §2, Fig 1).
_DECODER_WIDTHS: tuple[int, int, int, int, int] = (504, 224, 192, 96, 64)

_NUM_CLASSES: int = 3


class DenseUNet3d(nn.Module):
    """3D-DenseUNet-569: input ``(N, 1, 12, 224, 224)`` -> ``(N, 3, 12, 224, 224)``."""

    def __init__(self) -> None:
        super().__init__()

        # ---- Encoder ---------------------------------------------------
        # Stem: Conv3D k=7 s=2 pad=3 -> 96. 224->112, depth 12->6.
        self.stem = nn.Sequential(
            nn.Conv3d(1, _STEM_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(_STEM_CHANNELS),
            nn.ReLU(inplace=True),
        )
        # Pool: MaxPool3D k=2 s=2. 112->56, depth 6->3.
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        c = _STEM_CHANNELS  # running channel width

        # Dense Block 1 -> Transition 1
        self.dense_block1 = DenseBlock(c, _BLOCK_COUNTS[0], _GROWTH)
        c = c + _BLOCK_COUNTS[0] * _GROWTH  # 96 + 2*32 = 160
        self.db1_channels = c
        self.transition1 = TransitionBlock(c, compression=_COMPRESSION)
        c = self.transition1.out_channels  # 80

        # Dense Block 2 -> Transition 2
        self.dense_block2 = DenseBlock(c, _BLOCK_COUNTS[1], _GROWTH)
        c = c + _BLOCK_COUNTS[1] * _GROWTH  # 80 + 6*32 = 272
        self.db2_channels = c
        self.transition2 = TransitionBlock(c, compression=_COMPRESSION)
        c = self.transition2.out_channels  # 136

        # Dense Block 3 -> Transition 3
        self.dense_block3 = DenseBlock(c, _BLOCK_COUNTS[2], _GROWTH)
        c = c + _BLOCK_COUNTS[2] * _GROWTH  # 136 + 12*32 = 520
        self.db3_channels = c
        self.transition3 = TransitionBlock(c, compression=_COMPRESSION)
        c = self.transition3.out_channels  # 260

        # Dense Block 4 (decoder bottom @ 7x7x3)
        self.dense_block4 = DenseBlock(c, _BLOCK_COUNTS[3], _GROWTH)
        c = c + _BLOCK_COUNTS[3] * _GROWTH  # 260 + 18*32 = 836
        self.db4_channels = c

        # ---- Decoder ---------------------------------------------------
        # up1: bottom (DB4) -> 14x14x3, skip = DB3 (520 @ 14x14x3)
        self.up1 = UpsamplingBlock(
            in_channels=self.db4_channels,
            skip_channels=self.db3_channels,
            out_channels=_DECODER_WIDTHS[0],
            target_size=(3, 14, 14),
        )
        # up2: -> 28x28x3, skip = DB2 (272 @ 28x28x3)
        self.up2 = UpsamplingBlock(
            in_channels=_DECODER_WIDTHS[0],
            skip_channels=self.db2_channels,
            out_channels=_DECODER_WIDTHS[1],
            target_size=(3, 28, 28),
        )
        # up3: -> 56x56x3, skip = DB1 (160 @ 56x56x3)
        self.up3 = UpsamplingBlock(
            in_channels=_DECODER_WIDTHS[1],
            skip_channels=self.db1_channels,
            out_channels=_DECODER_WIDTHS[2],
            target_size=(3, 56, 56),
        )
        # up4: -> 112x112x6, skip = stem (96 @ 112x112x6)
        self.up4 = UpsamplingBlock(
            in_channels=_DECODER_WIDTHS[2],
            skip_channels=_STEM_CHANNELS,
            out_channels=_DECODER_WIDTHS[3],
            target_size=(6, 112, 112),
        )
        # up5: -> 224x224x12, NO encoder skip (skip_channels=0)
        self.up5 = UpsamplingBlock(
            in_channels=_DECODER_WIDTHS[3],
            skip_channels=0,
            out_channels=_DECODER_WIDTHS[4],
            target_size=(12, 224, 224),
        )

        # Classifier: Conv3D 1x1x1 -> 3
        self.classifier = nn.Conv3d(_DECODER_WIDTHS[4], _NUM_CLASSES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(N, 1, 12, 224, 224)`` to ``(N, 3, 12, 224, 224)``."""
        # Encoder, retaining the 4 skip tensors.
        stem = self.stem(x)  # 96   @ 112x112x6
        pooled = self.pool(stem)  # 96   @ 56x56x3

        db1 = self.dense_block1(pooled)  # 160  @ 56x56x3
        t1 = self.transition1(db1)  # 80   @ 28x28x3
        db2 = self.dense_block2(t1)  # 272  @ 28x28x3
        t2 = self.transition2(db2)  # 136  @ 14x14x3
        db3 = self.dense_block3(t2)  # 520  @ 14x14x3
        t3 = self.transition3(db3)  # 260  @ 7x7x3
        db4 = self.dense_block4(t3)  # 836  @ 7x7x3

        # Decoder with U-Net skips (skip is at target resolution, not upsampled).
        d = self.up1(db4, db3)  # 504 @ 14x14x3
        d = self.up2(d, db2)  # 224 @ 28x28x3
        d = self.up3(d, db1)  # 192 @ 56x56x3
        d = self.up4(d, stem)  # 96  @ 112x112x6
        # up5 has no encoder skip; pass an empty (0-channel) tensor at target size.
        empty_skip = x.new_zeros((x.shape[0], 0, 12, 224, 224))
        d = self.up5(d, empty_skip)  # 64  @ 224x224x12

        return cast(torch.Tensor, self.classifier(d))  # 3 @ 224x224x12
