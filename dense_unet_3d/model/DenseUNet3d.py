import torch
from torch import nn

from dense_unet_3d.model.building_blocks.DenseBlock import DenseBlock
from dense_unet_3d.model.building_blocks.TransitionBlock import TransitionBlock
from dense_unet_3d.model.building_blocks.UpsamplingBlock import UpsamplingBlock


class DenseUNet3d(nn.Module):
    def __init__(self):
        """
        Create the layers for the model
        """
        super().__init__()
        # Initial Layers
        self.conv1 = nn.Conv3d(
            1, 96, kernel_size=(7, 7, 7), stride=2, padding=(3, 3, 3)
        )
        self.bn1 = nn.BatchNorm3d(96)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        # Dense Layers
        self.transition = TransitionBlock(32)
        self.dense1 = DenseBlock(96, 128, 32, 4)
        self.dense2 = DenseBlock(32, 128, 32, 12)
        self.dense3 = DenseBlock(32, 128, 32, 24)
        self.dense4 = DenseBlock(32, 32, 32, 36)

        # Upsampling Layers
        self.upsample1 = UpsamplingBlock(32 + 32, 504, size=(1, 2, 2))
        self.upsample2 = UpsamplingBlock(504 + 32, 224, size=(1, 2, 2))
        self.upsample3 = UpsamplingBlock(224 + 32, 192, size=(1, 2, 2))
        self.upsample4 = UpsamplingBlock(192 + 32, 96, size=(2, 2, 2))
        self.upsample5 = UpsamplingBlock(96 + 96, 64, size=(2, 2, 2))

        # Final output layer
        # Typo in the paper? Says stride = 0 but that's impossible
        self.conv_classifier = nn.Conv3d(64, 3, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        :param x:  image tensor
        :return:   output of the forward pass
        """
        residual1 = self.relu(self.bn1(self.conv1(x)))
        residual2 = self.dense1(self.maxpool1(residual1))
        residual3 = self.dense2(self.transition(residual2))
        residual4 = self.dense3(self.transition(residual3))
        output = self.dense4(self.transition(residual4))

        output = self.upsample1(output, output)
        output = self.upsample2(output, residual4)
        output = self.upsample3(output, residual3)
        output = self.upsample4(output, residual2)
        output = self.upsample5(output, residual1)

        output = self.conv_classifier(output)

        return output
