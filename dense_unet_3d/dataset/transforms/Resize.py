from typing import Tuple

import torch
import torch.nn.functional as F


class Resize:
    """
    Resizes a 3D image tensor to the provided size
    """

    def __init__(self, size: Tuple):
        """
        Initialize the Resize transform

        :param size:    tuple containing the desired output size of the 3D image
                        The dimensions should be in format (D x H x W)
        """
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Call logic for Resize

        :param img: image tensor with dimensions (B x D x H x W)
        :return:    resized 3D image tensor
        """
        return F.interpolate(
            img.unsqueeze(0), self.size, mode="trilinear", align_corners=True
        ).squeeze(0)
