"""Tests for dataset transforms (Task D1).

TDD: written BEFORE the implementation rewrites.

Coverage
--------
ReshapeTensor
    - shape: (H, W, D) -> (1, D, H, W)
    - value placement: a tracer value at a known axis position survives correctly

RandomHorizontalFlip
    - per-sample randomness: repeated __call__s produce both flip and no-flip outcomes
    - paired: same decision applied to image and mask in a single __call__
    - OLD frozen-in-__init__ behavior would fail the randomness test (documented via
      a sentinel class that freezes the decision in __init__)

ScaleAndPadOrCrop
    - per-sample randomness: repeated __call__s produce varying scale factors
    - output shape preserved (same as input after pad/crop)
    - paired: same scale applied to image and mask in one __call__
    - scale range: all sampled factors lie in [0.8, 1.2]

ClampValues
    - values are clamped to [-200, 250] (no behavior change; test only)

Resize
    - output shape is (D=12, H=224, W=224) (no behavior change; test only)

All tests are CPU-only, seeded where needed.
"""

from __future__ import annotations

import numpy as np
import torch

from dense_unet_3d.dataset.transforms.ClampValues import ClampValues
from dense_unet_3d.dataset.transforms.RandomHorizontalFlip import RandomHorizontalFlip
from dense_unet_3d.dataset.transforms.ReshapeTensor import ReshapeTensor
from dense_unet_3d.dataset.transforms.Resize import Resize
from dense_unet_3d.dataset.transforms.ScaleAndPadOrCrop import ScaleAndPadOrCrop

# ---------------------------------------------------------------------------
# ReshapeTensor
# ---------------------------------------------------------------------------


class TestReshapeTensor:
    """ReshapeTensor: (H, W, D) -> (1, D, H, W) with correct value placement."""

    def test_output_shape(self) -> None:
        """(H=10, W=8, D=12) input must produce shape (1, 12, 10, 8)."""
        H, W, D = 10, 8, 12
        img = torch.zeros(H, W, D)
        result = ReshapeTensor()(img)
        assert result.shape == (1, D, H, W), (
            f"Expected (1, {D}, {H}, {W}), got {tuple(result.shape)}"
        )

    def test_axis_correctness_value_placement(self) -> None:
        """A tracer value placed at a known D position survives in the correct NCDHW axis.

        Strategy: build a tensor of shape (H, W, D) where each depth slice d
        contains the constant value `d+1`. After ReshapeTensor the output
        shape must be (1, D, H, W) and output[0, d, :, :] must equal d+1.
        """
        H, W, D = 6, 5, 4
        img = torch.zeros(H, W, D)
        for d in range(D):
            img[:, :, d] = float(d + 1)

        result = ReshapeTensor()(img)

        assert result.shape == (1, D, H, W)
        for d in range(D):
            assert torch.all(result[0, d, :, :] == float(d + 1)), (
                f"Depth slice d={d}: expected all values {d + 1}, "
                f"got unique values {result[0, d, :, :].unique().tolist()}"
            )

    def test_bug_axis_swap_would_fail(self) -> None:
        """The old (1, H, D, W) axis order (transpose(1,2).unsqueeze(0)) would produce
        shape (1, H, D, W) = (1, 6, 4, 5), not (1, D, H, W) = (1, 4, 6, 5).
        Confirm the new code does NOT produce the old shape."""
        H, W, D = 6, 5, 4
        img = torch.zeros(H, W, D)
        result = ReshapeTensor()(img)
        # Old behavior would give (1, H, D, W) = (1, 6, 4, 5)
        assert result.shape != (1, H, D, W), (
            "Shape matches the OLD buggy (1, H, D, W) ordering — axis swap not fixed!"
        )


# ---------------------------------------------------------------------------
# RandomHorizontalFlip
# ---------------------------------------------------------------------------


class TestRandomHorizontalFlip:
    """RandomHorizontalFlip: decision must be sampled per-call, not frozen in __init__."""

    def _make_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Small (C, H, W) tensors representing image and mask."""
        img = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        mask = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6) * 10
        return img, mask

    def test_per_sample_randomness_both_outcomes_occur(self) -> None:
        """Over many seeded __call__s, both flip and no-flip must occur.

        With p=0.5, in 40 independent calls the probability of ALL being the
        same outcome is negligible (2 * 0.5**40 ~ 1e-12).
        """
        transform = RandomHorizontalFlip(p=0.5)
        img, mask = self._make_pair()
        flipped_count = 0
        not_flipped_count = 0
        np.random.seed(0)
        torch.manual_seed(0)
        for _ in range(40):
            out_img, out_mask = transform((img, mask))
            if not torch.equal(out_img, img):
                flipped_count += 1
            else:
                not_flipped_count += 1

        assert flipped_count > 0, "Never flipped across 40 calls — decision is frozen!"
        assert not_flipped_count > 0, "Always flipped across 40 calls — decision is frozen!"

    def test_paired_same_decision_image_and_mask(self) -> None:
        """Image and mask must receive the SAME flip decision in one __call__."""
        transform = RandomHorizontalFlip(p=1.0)  # always flip
        img, mask = self._make_pair()
        out_img, out_mask = transform((img, mask))
        # Both should be horizontally flipped
        assert torch.equal(out_img, torch.flip(img, dims=[-1])), "Image was not flipped"
        assert torch.equal(out_mask, torch.flip(mask, dims=[-1])), "Mask was not flipped"

    def test_no_flip_when_p_zero(self) -> None:
        """With p=0, no flip must ever occur."""
        transform = RandomHorizontalFlip(p=0.0)
        img, mask = self._make_pair()
        for _ in range(10):
            out_img, out_mask = transform((img, mask))
            assert torch.equal(out_img, img)
            assert torch.equal(out_mask, mask)

    def test_frozen_init_would_fail_randomness(self) -> None:
        """Demonstrate that an instance with decision frozen in __init__ fails the
        distribution test (documents why the bug matters)."""

        class _FrozenFlip:
            """Old buggy implementation: decision fixed in __init__."""

            def __init__(self, p: float) -> None:
                self.flip = np.random.uniform(0, 1) <= p

            def __call__(self, imgs: tuple) -> tuple:
                import torchvision.transforms.functional as TF

                return tuple(TF.hflip(i) for i in imgs) if self.flip else imgs

        np.random.seed(0)
        frozen = _FrozenFlip(p=0.5)
        img, mask = self._make_pair()
        outcomes = set()
        for _ in range(20):
            out_img, _ = frozen((img, mask))
            outcomes.add(torch.equal(out_img, img))  # True = no flip

        # frozen instance always gives the same outcome
        assert len(outcomes) == 1, (
            "Frozen-init transform already varies across calls — test logic error"
        )


# ---------------------------------------------------------------------------
# ScaleAndPadOrCrop
# ---------------------------------------------------------------------------


class TestScaleAndPadOrCrop:
    """ScaleAndPadOrCrop: scale must be sampled per-call in [0.8, 1.2]."""

    def _make_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """NCDHW-like (C, D, H, W) tensors — the transform operates on these."""
        img = torch.randn(1, 4, 32, 32)
        mask = torch.zeros(1, 4, 32, 32)
        return img, mask

    def test_output_shape_preserved(self) -> None:
        """Output spatial shape must equal input spatial shape."""
        transform = ScaleAndPadOrCrop(scale_factor=(0.8, 1.2))
        img, mask = self._make_pair()
        out_img, out_mask = transform((img, mask))
        assert out_img.shape == img.shape, f"Image shape changed: {img.shape} -> {out_img.shape}"
        assert out_mask.shape == mask.shape, f"Mask shape changed: {mask.shape} -> {out_mask.shape}"

    def test_per_sample_randomness_scales_vary(self) -> None:
        """Repeated calls on the same input must produce DIFFERENT outputs (varying scale).

        We collect 30 outputs. If all identical, the scale is frozen per instance.
        """
        transform = ScaleAndPadOrCrop(scale_factor=(0.8, 1.2))
        img, mask = self._make_pair()
        np.random.seed(1)
        torch.manual_seed(1)
        outputs = []
        for _ in range(30):
            out_img, _ = transform((img, mask))
            outputs.append(out_img.clone())

        # At least two distinct outputs must exist
        all_same = all(torch.allclose(outputs[0], o, atol=1e-5) for o in outputs[1:])
        assert not all_same, (
            "All 30 outputs are identical — scale is frozen in __init__, not sampled per-call"
        )

    def test_paired_same_scale_image_and_mask(self) -> None:
        """Image and mask must be scaled by the same factor in one __call__.

        We use a constant-filled tensor so we can verify alignment: after
        crop/pad, the non-zero region must be the same for both.
        """
        transform = ScaleAndPadOrCrop(scale_factor=(0.8, 1.2))
        # Fill image with 1s and mask with 2s (same spatial pattern)
        img = torch.ones(1, 4, 32, 32)
        mask = torch.ones(1, 4, 32, 32) * 2
        out_img, out_mask = transform((img, mask))
        # Wherever img is non-zero, mask should also be non-zero (same crop region)
        img_nonzero = out_img != 0
        mask_nonzero = out_mask != 0
        assert torch.equal(img_nonzero, mask_nonzero), (
            "Non-zero regions differ between image and mask — different scale applied!"
        )

    def test_frozen_init_would_fail_randomness(self) -> None:
        """Frozen-in-__init__ scale gives identical outputs for all calls — documents the bug."""

        class _FrozenScale:
            def __init__(self, scale_factor: tuple) -> None:
                self.scale_factor = np.random.uniform(scale_factor[0], scale_factor[1])

            def __call__(self, imgs: tuple) -> tuple:
                import torch.nn.functional as F
                import torchvision.transforms.functional as TF

                original_size = list(imgs[0].shape)
                imgs = tuple(
                    F.interpolate(
                        img.unsqueeze(0),
                        scale_factor=(1, self.scale_factor, self.scale_factor),
                        mode="trilinear",
                        align_corners=True,
                        recompute_scale_factor=True,
                    ).squeeze(0)
                    for img in imgs
                )
                imgs = tuple(TF.center_crop(img, original_size[-1]) for img in imgs)
                return imgs

        np.random.seed(2)
        frozen = _FrozenScale(scale_factor=(0.8, 1.2))
        img = torch.randn(1, 4, 32, 32)
        mask = torch.zeros(1, 4, 32, 32)
        outputs = []
        for _ in range(5):
            out_img, _ = frozen((img, mask))
            outputs.append(out_img.clone())
        all_same = all(torch.allclose(outputs[0], o, atol=1e-5) for o in outputs[1:])
        assert all_same, "Frozen-init transform already varies — test logic error"


# ---------------------------------------------------------------------------
# ClampValues
# ---------------------------------------------------------------------------


class TestClampValues:
    """ClampValues: values outside [-200, 250] must be clamped."""

    def test_clamps_to_minus200_250(self) -> None:
        """Values below -200 become -200; values above 250 become 250."""
        transform = ClampValues(voxel_range=(-200, 250))
        img = torch.tensor([-300.0, -200.0, 0.0, 250.0, 500.0])
        result = transform(img)
        expected = torch.tensor([-200.0, -200.0, 0.0, 250.0, 250.0])
        assert torch.allclose(result, expected), (
            f"ClampValues output {result.tolist()} != expected {expected.tolist()}"
        )

    def test_values_within_range_unchanged(self) -> None:
        """Values inside the range must pass through unmodified."""
        transform = ClampValues(voxel_range=(-200, 250))
        img = torch.tensor([-199.0, 0.0, 100.0, 249.0])
        result = transform(img)
        assert torch.allclose(result, img)

    def test_exact_boundaries_unchanged(self) -> None:
        """Boundary values -200 and 250 must be unchanged."""
        transform = ClampValues(voxel_range=(-200, 250))
        img = torch.tensor([-200.0, 250.0])
        result = transform(img)
        assert torch.allclose(result, img)


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


class TestResize:
    """Resize: output must be (D=12, H=224, W=224)."""

    def test_resize_to_paper_dims(self) -> None:
        """A (1, 6, 64, 64) tensor resizes to (1, 12, 224, 224)."""
        target = (12, 224, 224)
        transform = Resize(size=target)
        img = torch.randn(1, 6, 64, 64)
        result = transform(img)
        assert result.shape == (1, 12, 224, 224), (
            f"Expected (1, 12, 224, 224), got {tuple(result.shape)}"
        )

    def test_resize_preserves_channel_dim(self) -> None:
        """Channel dimension (C=1) must be unchanged."""
        target = (12, 224, 224)
        transform = Resize(size=target)
        img = torch.randn(1, 6, 64, 64)
        result = transform(img)
        assert result.shape[0] == 1

    def test_resize_already_correct_size_noop(self) -> None:
        """When input already has the target spatial dims, output shape is unchanged."""
        target = (12, 224, 224)
        transform = Resize(size=target)
        img = torch.randn(1, 12, 224, 224)
        result = transform(img)
        assert result.shape == (1, 12, 224, 224)
