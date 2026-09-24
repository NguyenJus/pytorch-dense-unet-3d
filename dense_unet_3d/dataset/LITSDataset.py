"""LiTS Dataset: loads NIfTI volume+segmentation pairs via nibabel.

Labels: 0 = background, 1 = liver, 2 = tumour/lesion (as in the paper).
"""

from __future__ import annotations

import os
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
from nibabel.spatialimages import SpatialImage
from torch.utils.data import Dataset


def _case_id(path: str, prefix: str) -> str:
    """Return the LiTS case suffix from a ``volume``/``segmentation`` path."""
    name = os.path.basename(path)
    for extension in (".nii.gz", ".nii"):
        if name.endswith(extension):
            name = name[: -len(extension)]
            break
    return name[len(prefix) :]


def discover_pairs(img_dirs: list[str]) -> list[tuple[str, str]]:
    """Discover and validate one segmentation for every volume, by case ID.

    Sorting the two glob results independently is unsafe: a missing case shifts
    every following image/mask pairing.  Pair by their shared filename suffix
    instead and reject incomplete or duplicate datasets before training starts.
    """
    volumes: dict[str, str] = {}
    segmentations: dict[str, str] = {}
    for directory in img_dirs:
        if not isinstance(directory, str) or not directory:
            raise ValueError("dataset directories must be non-empty paths")
        try:
            entries = os.listdir(directory)
        except OSError as exc:
            raise ValueError(f"cannot read dataset directory: {directory}") from exc
        for name in entries:
            path = os.path.join(directory, name)
            if not os.path.isfile(path) or not (name.endswith(".nii") or name.endswith(".nii.gz")):
                continue
            if name.startswith("volume"):
                case_id = _case_id(path, "volume")
                if case_id in volumes:
                    raise ValueError(f"duplicate volume case ID {case_id!r}")
                volumes[case_id] = path
            elif name.startswith("segmentation"):
                case_id = _case_id(path, "segmentation")
                if case_id in segmentations:
                    raise ValueError(f"duplicate segmentation case ID {case_id!r}")
                segmentations[case_id] = path

    missing_segmentations = sorted(set(volumes) - set(segmentations))
    missing_volumes = sorted(set(segmentations) - set(volumes))
    if missing_segmentations or missing_volumes:
        raise ValueError(
            "volume/segmentation case IDs do not match "
            f"(missing segmentations: {missing_segmentations}; "
            f"missing volumes: {missing_volumes})"
        )
    return [(volumes[case_id], segmentations[case_id]) for case_id in sorted(volumes)]


class LITSDataset(Dataset):
    """PyTorch Dataset for the Liver Tumour Segmentation (LiTS) challenge.

    Each item is a ``(image, mask)`` pair where:
    - ``image`` is a float32 tensor of shape ``(1, D, H, W)`` (channel-first).
    - ``mask`` is a long (int64) tensor with integer labels in {0, 1, 2}.

    Parameters
    ----------
    img_dirs:
        Directories to scan for ``volume*.nii`` and ``segmentation*.nii`` files.
    detect_tumors:
        When *True* (default) labels 0/1/2 are preserved.  When *False*,
        label 2 is collapsed to 1 (phase-1 liver-only training).
    crop_to_liver:
        When *True*, depth slices that contain no liver/tumour voxels are
        removed before transforms are applied.
    transform:
        Optional callable applied to the **image** numpy array after the
        (H, W, D) → (D, H, W) transpose, before tensor conversion.
    paired_transform:
        Optional callable applied to ``(image_tensor, mask_tensor)`` pairs —
        used for random augmentations that must be identical on both.
    """

    def __init__(
        self,
        img_dirs: list[str],
        detect_tumors: bool | None = True,
        crop_to_liver: bool | None = False,
        transform: Any | None = None,
        mask_transform: Any | None = None,
        paired_transform: Any | None = None,
    ) -> None:
        pairs = discover_pairs(img_dirs)
        self.volume_img_paths = [volume for volume, _segmentation in pairs]
        self.segmentation_img_paths = [segmentation for _volume, segmentation in pairs]

        self.transform = transform
        # Mask-specific transform path (nearest-neighbour resize, no HU clamp).
        # Falls back to ``transform`` only if no dedicated mask path is given.
        self.mask_transform = mask_transform
        self.paired_transform = paired_transform
        self.detect_tumors = detect_tumors
        self.crop_to_liver = crop_to_liver

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def find_liver(self, imgs: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Crop the depth axis to slices that contain foreground voxels.

        Parameters
        ----------
        imgs:
            ``(vol, seg)`` pair with shape ``(D, H, W)`` each.

        Returns
        -------
        vol, seg:
            Pair of arrays with shape ``(H, W, n_slices)`` where *n_slices*
            is the count of depth slices that have at least one non-zero
            segmentation voxel.

        Note
        ----
        The original implementation contained ``return tuple(vol, seg)`` which
        raises ``TypeError`` because :func:`tuple` accepts at most one argument.
        The fix is the plain comma-separated return ``return vol, seg``.
        """
        vol_arr, seg_arr = imgs
        depth = seg_arr.shape[0]

        n_slice = [i for i in range(depth) if seg_arr[i].sum() > 0]

        if not n_slice:
            raise ValueError(
                "find_liver: segmentation volume contains no foreground "
                "(liver/tumour) voxels — every depth slice is all-background, "
                "so there is nothing to crop to. All-background volumes are "
                "valid in LiTS but cannot be used with crop_to_liver=True; "
                "filter them out or disable crop_to_liver."
            )

        vol_cropped = np.transpose(np.array([vol_arr[i] for i in n_slice]), (1, 2, 0))
        seg_cropped = np.transpose(np.array([seg_arr[i] for i in n_slice]), (1, 2, 0))

        # Fix: was ``return tuple(vol, seg)`` — TypeError; correct form is plain return.
        return vol_cropped, seg_cropped

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of volume-segmentation pairs in the dataset."""
        return len(self.volume_img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return the ``(image, mask)`` pair at *idx*.

        Returns
        -------
        image:
            Float32 tensor of shape ``(1, D, H, W)`` (channel-first CDHW).
        mask:
            Long (int64) tensor with labels in {0, 1, 2}.
        """
        # Load NIfTI — stored as (H, W, D) per NIfTI convention.
        # Cast to SpatialImage so mypy knows get_fdata() is available.
        vol_img = cast(SpatialImage, nib.load(self.volume_img_paths[idx]))
        seg_img = cast(SpatialImage, nib.load(self.segmentation_img_paths[idx]))
        volume: np.ndarray = np.asarray(vol_img.get_fdata(), dtype=np.float32)
        segmentation: np.ndarray = np.asarray(seg_img.get_fdata(), dtype=np.float32)
        if volume.shape != segmentation.shape:
            raise ValueError(
                f"image/mask shape mismatch for {self.volume_img_paths[idx]}: "
                f"{volume.shape} != {segmentation.shape}"
            )
        if not np.allclose(vol_img.affine, seg_img.affine, rtol=0.0, atol=1e-5):
            raise ValueError(
                f"image/mask affine mismatch for {self.volume_img_paths[idx]}; "
                "they are not on the same spatial grid"
            )
        if not np.all(np.isclose(segmentation, np.round(segmentation))) or not np.all(
            (segmentation >= 0) & (segmentation <= 2)
        ):
            raise ValueError(
                f"segmentation labels for {self.segmentation_img_paths[idx]} must be integers in {{0, 1, 2}}"
            )

        # Keep NIfTI's HWD order until ReshapeTensor converts it once to CDHW.
        # Cropping operates depth-first, so temporarily transpose only for it.
        vol_arr: np.ndarray = volume
        seg_arr: np.ndarray = segmentation

        if self.crop_to_liver:
            vol_arr, seg_arr = self.find_liver(
                (np.transpose(vol_arr, (2, 0, 1)), np.transpose(seg_arr, (2, 0, 1)))
            )

        # Apply per-array transforms (may return ndarray or Tensor).
        vol_out: Any = vol_arr
        seg_out: Any = seg_arr
        if self.transform:
            vol_out = self.transform(vol_out)
        # The mask MUST use its own nearest-neighbour pipeline so integer
        # labels are never averaged by trilinear interpolation, and is never
        # HU-clamped.  Fall back to ``transform`` only if no mask path exists.
        seg_transform = self.mask_transform if self.mask_transform is not None else self.transform
        if seg_transform:
            seg_out = seg_transform(seg_out)

        # Convert to tensors if not already done by transforms.
        if not isinstance(vol_out, torch.Tensor):
            vol_out = torch.from_numpy(vol_out).permute(2, 0, 1).unsqueeze(0)
        if not isinstance(seg_out, torch.Tensor):
            seg_out = torch.from_numpy(seg_out).permute(2, 0, 1).unsqueeze(0)

        # Ensure float32 image, long mask.
        image: torch.Tensor = vol_out.float()
        mask: torch.Tensor = seg_out

        if self.paired_transform:
            image, mask = self.paired_transform((image, mask))

        # Round in case any spatial transform introduced interpolation artefacts.
        mask = torch.round(mask).long()

        # Phase-1 training: collapse tumour label → liver.
        if not self.detect_tumors:
            mask = torch.clamp(mask, 0, 1)

        # Ensure channel-first: (D, H, W) → (1, D, H, W).
        if image.dim() == 3:
            image = image.unsqueeze(0)

        return image, mask
