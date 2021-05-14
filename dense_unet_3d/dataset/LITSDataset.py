import glob
import os
from typing import List, Tuple, Optional, Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class LITSDataset(Dataset):
    """
    Class for LiTS Dataset
    """

    def __init__(
        self,
        img_dirs: List[str],
        detect_tumors: Optional[bool] = True,
        crop_to_liver: Optional[bool] = False,
        transform: Optional[Any] = None,
        paired_transform: Optional[Any] = None,
    ):
        """
        Initialize the LiTS Dataset

        :param img_dirs:            list of image directories to pull images from
        :param detect_tumors:       boolean to tell whether to add a tumor class to the dataset
                                    If false, all tumors are treated as livers.
        :param transform:           list of transforms to conduct on volumes and segmentations
        :param paired_transform:    list of transforms which must be done on pairs of data
                                    This is used for randomized transforms which must be done
                                    the same way for a volume and its respective segmentation.
        """
        self.volume_img_paths = []
        self.segmentation_img_paths = []
        for path in img_dirs:
            self.volume_img_paths.extend(glob.glob(os.path.join(path, "volume*.nii")))
            self.segmentation_img_paths.extend(
                glob.glob(os.path.join(path, "segmentation*.nii"))
            )

        self.transform = transform
        self.paired_transform = paired_transform
        self.detect_tumors = detect_tumors
        self.crop_to_liver = crop_to_liver

    def find_liver(self, imgs: Tuple) -> Tuple:
        """
        Crops volume depth to the region where liver segmentations exist

        :param imgs:    volume-segmentation pair
        :return:        volume and segmentation cropped such that the first and last slices have liver detections
        """
        n_slice = []
        seg_img = imgs[1]
        depth = imgs[1].shape[0]

        for eachslice in np.arange(depth):
            if seg_img[eachslice].sum() > 0:
                n_slice.append(eachslice)
        vol = np.transpose(np.array([imgs[0][i] for i in n_slice]), (1, 2, 0))
        seg = np.transpose(np.array([imgs[1][i] for i in n_slice]), (1, 2, 0))

        return tuple(vol, seg)

    def __len__(self) -> int:
        """
        Get the length of the dataset

        :return:    length of the dataset
        """
        return len(self.volume_img_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a volume and segmentation pair given an index

        :param idx: index of the volume-segmentation pair
        :return:    tuple containing the volume-segmentation pair
        """
        volume = nib.load(self.volume_img_paths[idx]).get_fdata()
        volume = np.asarray(volume)

        segmentation = nib.load(self.segmentation_img_paths[idx]).get_fdata()
        segmentation = np.asarray(segmentation)

        vol_temp = np.transpose(volume, (2, 0, 1))
        seg_temp = np.transpose(segmentation, (2, 0, 1))

        if self.crop_to_liver:
            volume, segmentation = self.find_liver((vol_temp, seg_temp))

        if self.transform:
            volume = self.transform(volume)
            segmentation = self.transform(segmentation)

        if self.paired_transform:
            volume, segmentation = self.paired_transform((volume, segmentation))

        # In case any transforms modifies the segmentation values (namely resize)
        segmentation = torch.round(segmentation)

        # Phase 1 of training only detects liver regions
        if not self.detect_tumors:
            segmentation = torch.clamp(segmentation, 0, 1)

        return volume, segmentation
