"""LiTS Dataset: loads NIfTI volume+segmentation pairs via nibabel.

Labels: 0 = background, 1 = liver, 2 = tumour/lesion (as in the paper).
"""

from __future__ import annotations

import glob
import os
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
from nibabel.spatialimages import SpatialImage
from torch.utils.data import Dataset


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
        self.volume_img_paths: list[str] = []
        self.segmentation_img_paths: list[str] = []
        for path in img_dirs:
            self.volume_img_paths.extend(sorted(glob.glob(os.path.join(path, "volume*.nii"))))
            self.segmentation_img_paths.extend(
                sorted(glob.glob(os.path.join(path, "segmentation*.nii")))
            )

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

        # Reorder to (D, H, W) for depth-first processing.
        vol_arr: np.ndarray = np.transpose(volume, (2, 0, 1))
        seg_arr: np.ndarray = np.transpose(segmentation, (2, 0, 1))

        if self.crop_to_liver:
            # find_liver returns (H, W, D) arrays.
            vol_arr, seg_arr = self.find_liver((vol_arr, seg_arr))
            # Bring back to (D, H, W).
            vol_arr = np.transpose(vol_arr, (2, 0, 1))
            seg_arr = np.transpose(seg_arr, (2, 0, 1))

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
            vol_out = torch.from_numpy(vol_out)
        if not isinstance(seg_out, torch.Tensor):
            seg_out = torch.from_numpy(seg_out)

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
