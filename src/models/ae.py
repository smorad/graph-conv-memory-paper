from torchvision.models import resnext50_32x4d
from torchvision.models.resnet import ResNet
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

import torch
from torch import nn


class CNNAutoEncoder(torch.nn.Module):
    def __init__(self, num_cats=42):
        super().__init__()
        self.num_cats = num_cats
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def loss(self, output, target):
        # Our input is 80 semantic layers and 1 depth layer
        # let's weight s.t. semantic and depth observations
        # are equal importance
        depth_e = torch.mean(output[:, 0] - target[:, 0]) ** 2
        semantic_e = torch.mean(output[:, 1:] - target[:, 1:] / self.num_cats) ** 2
        loss = depth_e + semantic_e
        return loss


class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 320x320
        self.encoder = nn.Sequential(
            nn.Conv2d(43 + 1, 64, 5, stride=3),  # b, 64, 10
            nn.ReLU(),
            nn.Conv2d(64, 96, 4, stride=2),  # b, 96, 4, 4
            nn.ReLU(),
            nn.Conv2d(96, 128, 2, stride=2),  # b, 128, 2, 2
            nn.ReLU(),
        )

    def forward(self, x):
        """
        for layer in self.encoder:
            x = layer(x)
            print(x.shape)
        return x
        """
        return self.encoder(x)


class CNNDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 2, stride=2),  # b, 128, 2, 2
            nn.ReLU(),
            nn.ConvTranspose2d(96, 64, 4, stride=2),  # b, 96, 4, 4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 43 + 1, 5, stride=3),  # b, 64, 10
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        for layer in self.decoder:
            x = layer(x)
            print(x.shape)
        return x
        """
        return self.decoder(x)
