import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ScaleAndPadOrCrop:
    """Scale a tuple of tensors then center-crop or zero-pad back to the original size.

    A random scale factor in [*scale_lo*, *scale_hi*] is drawn fresh on every
    ``__call__`` (per-sample randomness).  All tensors in the tuple share the
    same scale so image/mask spatial alignment is preserved.

    Bug fixed: the previous implementation drew the scale factor once in
    ``__init__``, freezing it across all calls on that instance.  This meant
    every sample processed by the same transform instance was scaled identically
    — not true per-sample augmentation.

    The scale is applied only to the in-plane (H, W) dimensions (axis -1 and
    -2).  The depth axis (D) is kept at its original size to match the paper's
    augmentation scheme.
    """

    def __init__(self, scale_factor: tuple[float, float]) -> None:
        """Initialise the transform.

        :param scale_factor: (lo, hi) bounds for the uniform scale distribution.
                             The paper uses (0.8, 1.2).
        """
        self.scale_lo: float = scale_factor[0]
        self.scale_hi: float = scale_factor[1]

    def __call__(self, imgs: tuple) -> tuple:
        """Scale all tensors in *imgs* by a freshly-sampled factor then crop/pad.

        :param imgs: tuple of float32 tensors, all the same shape (C, D, H, W).
        :return:     tuple of tensors with the same shape as the inputs.
        """
        scale: float = np.random.uniform(self.scale_lo, self.scale_hi)
        original_size = list(imgs[0].shape)
        imgs = tuple(
            F.interpolate(
                img.unsqueeze(0),
                scale_factor=(1, scale, scale),
                mode="trilinear",
                align_corners=True,
                recompute_scale_factor=True,
            ).squeeze(0)
            for img in imgs
        )
        imgs = tuple(TF.center_crop(img, original_size[-1]) for img in imgs)
        return imgs
