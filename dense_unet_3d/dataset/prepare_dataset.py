from typing import Dict, Optional

from torch.utils.data import DataLoader
from torchvision import transforms

from dense_unet_3d.dataset.LITSDataset import LITSDataset
from dense_unet_3d.dataset.transforms.ClampValues import ClampValues
from dense_unet_3d.dataset.transforms.RandomHorizontalFlip import RandomHorizontalFlip
from dense_unet_3d.dataset.transforms.ReshapeTensor import ReshapeTensor
from dense_unet_3d.dataset.transforms.Resize import Resize
from dense_unet_3d.dataset.transforms.ScaleAndPadOrCrop import ScaleAndPadOrCrop


def compose_transforms(config: Dict) -> Dict:
    """
    Composes the necessary transforms into lists based on user configuration

    :param config:  dictionary containing configuration instructions
    :return:        dictionary containing lists of transforms and paired transforms
    """
    all_transforms = [
        transforms.ToTensor(),
        ReshapeTensor(),
    ]

    # Transforms that must be completed on a set of images
    # These are usually probability-based transforms which must happen on both volume and segmentation
    paired_transforms = []

    dataset_configs = config["dataset"]

    if dataset_configs["clamp_hu"]:
        min_hu = dataset_configs["clamp_hu_range"]["min"]
        max_hu = dataset_configs["clamp_hu_range"]["max"]
        all_transforms.append(ClampValues((min_hu, max_hu)))

    if dataset_configs["resize_img"]:
        dims = dataset_configs["resize_dims"]
        img_size = (dims["D"], dims["H"], dims["W"])
        all_transforms.append(Resize(img_size))

    if dataset_configs["random_hflip"]:
        probability = dataset_configs["random_hflip_probability"]
        paired_transforms.append(RandomHorizontalFlip(probability))

    if dataset_configs["scale_img"]:
        min_scale = dataset_configs["scale_img_range"]["min"]
        max_scale = dataset_configs["scale_img_range"]["max"]
        paired_transforms.append(ScaleAndPadOrCrop((min_scale, max_scale)))

    return {
        "all_transforms": transforms.Compose(all_transforms),
        "paired_transforms": transforms.Compose(paired_transforms),
    }


def prepare_dataset(config: Dict, train: bool) -> LITSDataset:
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

    transform = compose_transforms(config)
    all_transforms = transform["all_transforms"]
    paired_transforms = transform["paired_transforms"]

    dataset = LITSDataset(
        img_dirs,
        transform=all_transforms,
        paired_transform=paired_transforms,
    )

    return dataset


def prepare_dataloader(config: Dict, train: Optional[bool] = True) -> DataLoader:
    """
    Builds the dataloader class to pass into PyTorch

    :param config:  dictionary containing configuration instructions
    :param train:   boolean to tell whether to use train or test images
    :return:        DataLoader class with dataset loaded
    """
    dataset = prepare_dataset(config, train)
    batch_size = config["dataset"]["batch_size"]
    shuffle = config["dataset"]["shuffle"]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader
