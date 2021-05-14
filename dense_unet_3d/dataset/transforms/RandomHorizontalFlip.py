from typing import Tuple

import numpy as np
import torchvision.transforms.functional as TF


class RandomHorizontalFlip:
    """
    Flip a tuple of tensors across a vertical axis with p probability
    """

    def __init__(self, p: float):
        """
        Initialize the RandomHorizontalFlip transform

        :param p:   probability of flipping in range [0, 1]
        """
        self.flip = True if np.random.uniform(0, 1) <= p else False

    def __call__(self, imgs: Tuple) -> Tuple:
        """
        Call logic for RandomHorizontalFlip

        :param imgs:    tuple of image tensors to flip together
                        By probability p, all tensors will flip
        :return:        tuple of images flipped by probability p
        """
        return tuple(TF.hflip(img) for img in imgs) if self.flip else imgs
