import os
from typing import Any

from torch.utils.data import DataLoader
from torchvision import transforms

from dense_unet_3d.dataset.LITSDataset import LITSDataset, _case_id, discover_pairs, preflight_pairs
from dense_unet_3d.dataset.transforms.ClampValues import ClampValues
from dense_unet_3d.dataset.transforms.RandomHorizontalFlip import RandomHorizontalFlip
from dense_unet_3d.dataset.transforms.ReshapeTensor import ReshapeTensor
from dense_unet_3d.dataset.transforms.Resize import Resize
from dense_unet_3d.dataset.transforms.ScaleAndPadOrCrop import ScaleAndPadOrCrop


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
    all_transforms: list[Any] = [
        ReshapeTensor(),
    ]

    # Mask pipeline: same tensor/reshape steps, but NO HU clamp and a
    # nearest-neighbour resize so labels stay integer.
    mask_transforms: list[Any] = [
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


def preflight_config(config: dict, *, full_decode: bool = False) -> dict[str, int]:
    """Validate every configured training and validation pair before training.

    This deliberately operates on raw NIfTI pairs, before data loaders, models,
    or CUDA are created.  ``full_decode`` adds segmentation-label validation;
    header-only checks are useful for a fast standalone audit.
    """
    pathing = config["pathing"]
    train_dirs = pathing.get("train_img_dirs")
    test_dirs = pathing.get("test_img_dirs")
    if (
        not isinstance(train_dirs, list)
        or not train_dirs
        or any(not isinstance(directory, str) or not directory for directory in train_dirs)
    ):
        raise ValueError("pathing.train_img_dirs must contain one or more non-empty directories")
    if (
        not isinstance(test_dirs, list)
        or not test_dirs
        or any(not isinstance(directory, str) or not directory for directory in test_dirs)
    ):
        raise ValueError("pathing.test_img_dirs must contain one or more non-empty directories")

    splits = (
        ("train", "training split", train_dirs),
        ("validation", "validation split", test_dirs),
    )
    discovered: dict[str, list[tuple[str, str]]] = {}
    counts: dict[str, int] = {}
    errors: list[str] = []
    for key, split_name, directories in splits:
        try:
            discovered[key] = discover_pairs(directories)
        except ValueError as exc:
            errors.append(f"{split_name} discovery failed: {exc}")

    if "train" in discovered and "validation" in discovered:
        train_pairs = discovered["train"]
        test_pairs = discovered["validation"]
        train_paths = {os.path.realpath(volume) for volume, _segmentation in train_pairs}
        test_paths = {os.path.realpath(volume) for volume, _segmentation in test_pairs}
        train_case_ids = {_case_id(volume, "volume") for volume, _segmentation in train_pairs}
        test_case_ids = {_case_id(volume, "volume") for volume, _segmentation in test_pairs}
        if train_paths & test_paths or train_case_ids & test_case_ids:
            errors.append(
                "train_img_dirs and test_img_dirs overlap; validation data would leak into training"
            )

    for key, split_name, _directories in splits:
        if key not in discovered:
            continue
        try:
            counts[key] = preflight_pairs(
                discovered[key], full_decode=full_decode, split_name=split_name
            )
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("configured-split preflight failed:\n" + "\n".join(errors))
    return counts


def prepare_dataset(
    config: dict,
    train: bool,
    *,
    detect_tumors: bool = True,
) -> LITSDataset:
    """
    Builds the dataset based on user configuration

    :param config:  dictionary containing configuration instructions
    :param train:   boolean to tell whether to pull training or testing images
    :param detect_tumors: when ``False``, collapse tumour label 2 to liver
        label 1 for the Phase A liver-only task.
    :return:        a created LITSDataset class
    """
    if train:
        img_dirs = config["pathing"]["train_img_dirs"]
    else:
        img_dirs = config["pathing"].get("test_img_dirs")
        if (
            img_dirs is None
            or len(img_dirs) == 0
            or any(not isinstance(d, str) or not d for d in img_dirs)
        ):
            raise ValueError(
                "pathing.test_img_dirs must point to labeled volume directories for "
                "non-dry-run evaluation, but it is unset or contains null entries. "
                "Set pathing.test_img_dirs in your config to one or more valid directories."
            )
        train_dirs = config["pathing"].get("train_img_dirs")
        if train_dirs:
            train_pairs = discover_pairs(train_dirs)
            test_pairs = discover_pairs(img_dirs)
            train_paths = {os.path.realpath(volume) for volume, _segmentation in train_pairs}
            test_paths = {os.path.realpath(volume) for volume, _segmentation in test_pairs}
            train_case_ids = {_case_id(volume, "volume") for volume, _segmentation in train_pairs}
            test_case_ids = {_case_id(volume, "volume") for volume, _segmentation in test_pairs}
            if train_paths & test_paths or train_case_ids & test_case_ids:
                raise ValueError(
                    "train_img_dirs and test_img_dirs overlap; validation data would leak into training"
                )

    transform = compose_transforms(config, train=train)
    all_transforms = transform["all_transforms"]
    mask_transforms = transform["mask_transforms"]
    paired_transforms = transform["paired_transforms"]

    dataset = LITSDataset(
        img_dirs,
        detect_tumors=detect_tumors,
        transform=all_transforms,
        mask_transform=mask_transforms,
        paired_transform=paired_transforms,
    )

    return dataset


def prepare_dataloader(
    config: dict,
    train: bool = True,
    *,
    detect_tumors: bool = True,
) -> DataLoader:
    """
    Builds the dataloader class to pass into PyTorch

    :param config:  dictionary containing configuration instructions
    :param train:   boolean to tell whether to use train or test images
    :param detect_tumors: when ``False``, return liver-only labels for Phase A.
    :return:        DataLoader class with dataset loaded
    """
    dataset = prepare_dataset(config, train, detect_tumors=detect_tumors)
    batch_size = config["dataset"]["batch_size"]
    # Never shuffle validation/test data — keeps evaluation deterministic.
    shuffle = config["dataset"]["shuffle"] if train else False

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader
