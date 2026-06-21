import torch


class ReshapeTensor:
    """Reshape a 3-D image tensor from (H, W, D) to NCDHW (1, D, H, W).

    NIfTI volumes loaded via nibabel arrive in (H, W, D) order.  The model
    expects NCDHW = (N, C, D, H, W).  This transform adds the channel (N=1)
    dimension and reorders the spatial axes so depth comes first.

    Bug fixed: the previous implementation used ``transpose(1, 2).unsqueeze(0)``
    which swapped axes 1 and 2 of (H, W, D) to give (H, D, W) and then
    unsqueezed to (1, H, D, W) — incorrect NCDHW ordering.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Reshape ``img`` from (H, W, D) to (1, D, H, W).

        :param img: float32 tensor with shape (H, W, D).
        :return:    float32 tensor with shape (1, D, H, W) — NCDHW.
        """
        # img: (H, W, D)  ->  permute to (D, H, W)  ->  unsqueeze to (1, D, H, W)
        return img.permute(2, 0, 1).unsqueeze(0)
