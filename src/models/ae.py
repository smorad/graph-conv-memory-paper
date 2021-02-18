from torchvision.models import resnext50_32x4d
from torchvision.models.resnet import ResNet
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

import torch
from torch import nn


class CNNAutoEncoder(torch.nn.Module):
    def __init__(self, num_cats=44):
        super().__init__()
        self.num_cats = num_cats
        self.h_dim = 2048
        # Takes 320x320
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
            nn.Unflatten(1, (self.h_dim // 4, 2, 2)),  # TODO: This should be variable
            nn.ConvTranspose2d(512, 256, 3, stride=1),  # b, 128, 2, 2
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2),  # b, 96, 4, 4
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
