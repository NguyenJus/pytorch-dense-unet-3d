from typing import Tuple

import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ScaleAndPadOrCrop:
    """
    Scale a tuple of tensors by a scale factor, then random crop or zero pad to the original size.
    This function expects all tensors to be the same shape.
    """

    def __init__(self, scale_factor: Tuple):
        """
        Initialize the ScaleAndPadOrCrop transform

        :param scale_factor:    tuple containing lower and upper bounds for the scaling of the image
        """
        self.scale_factor = np.random.uniform(scale_factor[0], scale_factor[1])

    def __call__(self, imgs: Tuple) -> Tuple:
        """
        Call logic for ScaleAndPadOrCrop

        :param imgs:    tuple of image tensors to scaled and padded or cropped together
                        All images in the tuple are scaled by the same factor
        :return:        tuple of scaled images
        """
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
