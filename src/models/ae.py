from torchvision.models import resnext50_32x4d
from torchvision.models.resnet import ResNet
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

import torch
from torch import nn


class CNNAutoEncoder(torch.nn.Module):
    def __init__(self, h_dim=2048):
        assert h_dim in [1024, 2048]
        super().__init__()
        self.h_dim = h_dim
        # Takes 32x32x44
        if h_dim == 2048:
            self.encoder = nn.Sequential(
                nn.Conv2d(43 + 1, 128, 5, stride=3),  # b, 64, 10
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, stride=2),  # b, 96, 4, 4
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, stride=1),  # b, 128, 2, 2
                nn.ReLU(),
                nn.Flatten(),
            )
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (self.h_dim // 4, 2, 2)),
                nn.ConvTranspose2d(512, 256, 3, stride=1),  # b, 128, 2, 2
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, stride=2),  # b, 96, 4, 4
                nn.ReLU(),
                nn.ConvTranspose2d(128, 43 + 1, 5, stride=3),  # b, 64, 10
                nn.Sigmoid(),
            )
        elif h_dim == 1024:
            self.encoder = nn.Sequential(
                nn.Conv2d(43 + 1, 128, 5, stride=3),  # b, 64, 10
                nn.ReLU(),
                nn.Conv2d(128, 196, 4, stride=2),  # b, 96, 4, 4
                nn.ReLU(),
                nn.Conv2d(196, 256, 3, stride=1),  # b, 128, 2, 2
                nn.ReLU(),
                nn.Flatten(),
            )
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (self.h_dim // 4, 2, 2)),
                nn.ConvTranspose2d(256, 196, 3, stride=1),  # b, 128, 2, 2
                nn.ReLU(),
                nn.ConvTranspose2d(196, 128, 4, stride=2),  # b, 96, 4, 4
                nn.ReLU(),
                nn.ConvTranspose2d(128, 43 + 1, 5, stride=3),  # b, 64, 10
                nn.Sigmoid(),
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decoder(self.encoder(x))
