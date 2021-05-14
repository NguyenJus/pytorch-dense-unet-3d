import yaml
import torch

from dense_unet_3d.dataset.prepare_dataset import prepare_dataloader
from dense_unet_3d.model.DenseUNet3d import DenseUNet3d
from dense_unet_3d.training.train import train
from dense_unet_3d.evaluation.evaluate import evaluate


def main():
    with open("./dense_unet_3d/config.yaml", "r") as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)

    if torch.cuda.is_available() and config["gpu"]["use_gpu"]:
        device = torch.device(config["gpu"]["gpu_name"])
    else:
        device = torch.device("cpu")

    trainloader = prepare_dataloader(config, train=True)

    model = DenseUNet3d()

    losses = train(config, model, device, trainloader)

    scores = evaluate(model, device, trainloader)
    print(scores)


if __name__ == "__main__":
    main()
