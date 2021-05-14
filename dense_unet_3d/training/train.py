import os

import torch
from torch import optim, nn
from tqdm import tqdm


def get_optimizer(model, config):
    optimizer = config["training"]["optimizer"]
    learning_rate = config["training"]["learning_rate"]

    if optimizer == "SGD":
        momentum = config["training"]["momentum"]
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )
    # TODO: implement first and second moments
    elif optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    return optimizer


def get_criterion(config):
    criterion = config["training"]["criterion"]

    if criterion == "CrossEntropyLoss":
        weights = config["training"]["class_weights"]
        class_weights = torch.tensor(
            [
                weights["background"],
                weights["liver"],
                weights["lesion"],
            ]
        )
        criterion = nn.CrossEntropyLoss(class_weights)
    # TODO: implement DiceLoss

    return criterion


def get_scheduler(optimizer, config):
    if not config["training"]["use_scheduler"]:
        return

    scheduler = config["training"]["scheduler"]
    if scheduler == "StepLR":
        step_size = config["training"]["scheduler_step"]
        gamma = config["training"]["scheduler_gamma"]
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    # TODO: implement other schedulers

    return scheduler


def train(config, model, device, dataloader):
    run_name = config["pathing"]["run_name"]
    model_save_dir = config["pathing"]["model_save_dir"]
    if not os.path.exists(f"./models/{run_name}"):
        os.makedirs(f"./models/{run_name}")

    model = model.to(device)
    optimizer = get_optimizer(model, config)
    criterion = get_criterion(config).to(device)
    scheduler = get_scheduler(optimizer, config)

    total_epochs = config["training"]["epochs"]
    starting_epoch = 0
    losses = []

    for epoch in tqdm(
        range(starting_epoch + 1, total_epochs + 1), position=0, leave=True
    ):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            volume, segmentation = data
            volume = volume.to(device, dtype=torch.float)
            segmentation = segmentation.to(device, dtype=torch.long)

            optimizer.zero_grad()

            output = model(volume)

            loss = criterion(output, segmentation.squeeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # evaluation
        model.eval()
        losses.append(running_loss / (i + 1))
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                    "losses": losses,
                },
                os.path.join(model_save_dir, run_name, f"epoch{epoch}.pt"),
            )

    return losses
