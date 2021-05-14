from typing import Tuple

import torch


class ClampValues:
    """
    Clamp voxels into the correct range
    Any values outside of the provided range are set as the min or max of that range
    """

    def __init__(self, voxel_range: Tuple):
        """
        Initialize the ClampValues transform

        :param voxel_range: Inclusive range of voxel values to clamp to

        """
        self.voxel_range = voxel_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Call logic for ClampValues

        :param img: image tensor with dimensions (B x D x H x W)
        :return:    image tensor with values clamped to the provided range
        """
        return torch.clamp(img, self.voxel_range[0], self.voxel_range[1])
