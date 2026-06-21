import math
import random

from torch.utils.data import DataLoader
from torchvision import transforms

from dense_unet_3d.dataset.LITSDataset import LITSDataset
from dense_unet_3d.dataset.transforms.ClampValues import ClampValues
from dense_unet_3d.dataset.transforms.RandomHorizontalFlip import RandomHorizontalFlip
from dense_unet_3d.dataset.transforms.ReshapeTensor import ReshapeTensor
from dense_unet_3d.dataset.transforms.Resize import Resize
from dense_unet_3d.dataset.transforms.ScaleAndPadOrCrop import ScaleAndPadOrCrop


def make_split(
    volume_ids: list[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return a deterministic (train, val) partition of *volume_ids*.

    The split is reproducible for a given *seed*: calling this function twice
    with the same arguments always produces identical lists.  The original
    list order is preserved after shuffling so the result depends only on the
    seed, not on any external randomness.

    Args:
        volume_ids:   Sequence of volume identifier strings to split.
        val_fraction: Fraction of volumes assigned to validation (default 0.2).
        seed:         Integer seed for the shuffle RNG.

    Returns:
        A ``(train_ids, val_ids)`` tuple of lists; both together form a
        partition of *volume_ids* (disjoint, union = all).
    """
    shuffled = list(volume_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_val = math.ceil(len(shuffled) * val_fraction)
    val_ids = shuffled[:n_val]
    train_ids = shuffled[n_val:]
    return train_ids, val_ids


def compose_transforms(config: dict, train: bool = True) -> dict:
    """
    Composes the necessary transforms into lists based on user configuration

    :param config:  dictionary containing configuration instructions
    :param train:   when *False*, random augmentations (RandomHorizontalFlip,
                    ScaleAndPadOrCrop) are dropped so validation/test is fully
                    deterministic; only resize/clamp/reshape remain.
    :return:        dictionary with three Compose objects:
                    - ``all_transforms``:  per-image (intensity) pipeline.
                    - ``mask_transforms``: per-mask pipeline — NEAREST resize,
                      no HU clamp, so integer labels are never averaged.
                    - ``paired_transforms``: random augmentations applied to
                      both image and mask together (train only).
    """
    # Intensity (volume) pipeline.
    all_transforms = [
        transforms.ToTensor(),
        ReshapeTensor(),
    ]

    # Mask pipeline: same tensor/reshape steps, but NO HU clamp and a
    # nearest-neighbour resize so labels stay integer.
    mask_transforms: list = [
        transforms.ToTensor(),
        ReshapeTensor(),
    ]

    # Transforms that must be completed on a set of images
    # These are usually probability-based transforms which must happen on both volume and segmentation
    paired_transforms: list[RandomHorizontalFlip | ScaleAndPadOrCrop] = []

    dataset_configs = config["dataset"]

    if dataset_configs["clamp_hu"]:
        min_hu = dataset_configs["clamp_hu_range"]["min"]
        max_hu = dataset_configs["clamp_hu_range"]["max"]
        all_transforms.append(ClampValues((min_hu, max_hu)))

    if dataset_configs["resize_img"]:
        dims = dataset_configs["resize_dims"]
        img_size = (dims["D"], dims["H"], dims["W"])
        all_transforms.append(Resize(img_size))
        # Mask resized with nearest-neighbour to preserve integer labels.
        mask_transforms.append(Resize(img_size, mode="nearest"))

    # Random augmentations apply to training only (deterministic validation).
    if train:
        if dataset_configs["random_hflip"]:
            probability = dataset_configs["random_hflip_probability"]
            paired_transforms.append(RandomHorizontalFlip(probability))

        if dataset_configs["scale_img"]:
            min_scale = dataset_configs["scale_img_range"]["min"]
            max_scale = dataset_configs["scale_img_range"]["max"]
            paired_transforms.append(ScaleAndPadOrCrop((min_scale, max_scale)))

    return {
        "all_transforms": transforms.Compose(all_transforms),
        "mask_transforms": transforms.Compose(mask_transforms),
        "paired_transforms": transforms.Compose(paired_transforms),
    }


def prepare_dataset(config: dict, train: bool) -> LITSDataset:
    """
    Builds the dataset based on user configuration

    :param config:  dictionary containing configuration instructions
    :param train:   boolean to tell whether to pull training or testing images
    :return:        a created LITSDataset class
    """
    if train:
        img_dirs = config["pathing"]["train_img_dirs"]
    else:
        img_dirs = config["pathing"]["test_img_dirs"]

    transform = compose_transforms(config, train=train)
    all_transforms = transform["all_transforms"]
    mask_transforms = transform["mask_transforms"]
    paired_transforms = transform["paired_transforms"]

    dataset = LITSDataset(
        img_dirs,
        transform=all_transforms,
        mask_transform=mask_transforms,
        paired_transform=paired_transforms,
    )

    return dataset


def prepare_dataloader(config: dict, train: bool = True) -> DataLoader:
    """
    Builds the dataloader class to pass into PyTorch

    :param config:  dictionary containing configuration instructions
    :param train:   boolean to tell whether to use train or test images
    :return:        DataLoader class with dataset loaded
    """
    dataset = prepare_dataset(config, train)
    batch_size = config["dataset"]["batch_size"]
    # Never shuffle validation/test data — keeps evaluation deterministic.
    shuffle = config["dataset"]["shuffle"] if train else False

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader
