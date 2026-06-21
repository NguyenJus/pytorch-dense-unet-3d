import torch
import torch.nn.functional as F


class Resize:
    """
    Resizes a 3D image tensor to the provided size
    """

    def __init__(self, size: tuple, mode: str = "trilinear"):
        """
        Initialize the Resize transform

        :param size:    tuple containing the desired output size of the 3D image
                        The dimensions should be in format (D x H x W)
        :param mode:    interpolation mode. Use ``"trilinear"`` (default) for
                        intensity volumes; use ``"nearest"`` for integer
                        segmentation masks so label values are never averaged.
        """
        self.size = size
        self.mode = mode

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Call logic for Resize

        :param img: image tensor with dimensions (B x D x H x W)
        :return:    resized 3D image tensor
        """
        # align_corners is only valid for linear/bilinear/trilinear modes;
        # it must be None for nearest-neighbour to avoid a runtime error.
        align_corners = True if self.mode == "trilinear" else None
        return F.interpolate(
            img.unsqueeze(0), self.size, mode=self.mode, align_corners=align_corners
        ).squeeze(0)
