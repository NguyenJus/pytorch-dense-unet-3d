import numpy as np
import torchvision.transforms.functional as TF


class RandomHorizontalFlip:
    """Randomly flip a tuple of tensors horizontally with probability *p*.

    The flip decision is sampled inside ``__call__`` so each invocation is
    independent (per-sample randomness).  All tensors in the tuple receive the
    same decision, ensuring paired image/mask consistency.

    Bug fixed: the previous implementation sampled the decision in ``__init__``,
    freezing it for the entire lifetime of the transform instance.  This meant
    every sample in a dataset would be either always flipped or never flipped
    depending on which instance was created — not true random augmentation.
    """

    def __init__(self, p: float) -> None:
        """Initialise the transform.

        :param p: probability of flipping, in range [0, 1].
        """
        self.p = p

    def __call__(self, imgs: tuple) -> tuple:
        """Apply a freshly-sampled flip decision to all tensors in *imgs*.

        :param imgs: tuple of tensors to (potentially) flip together.
        :return:     tuple of tensors, all flipped or all unchanged.
        """
        if np.random.uniform(0, 1) <= self.p:
            return tuple(TF.hflip(img) for img in imgs)
        return imgs
