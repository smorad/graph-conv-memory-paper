from torchvision.models import resnext50_32x4d
from torchvision.models.resnet import ResNet
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

import torch
from torch import nn


class ResNeXt50(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        self.model = resnext50_32x4d(pretrained=True)

    def forward(self, x):
        # https://pytorch.org/hub/pytorch_vision_resnext/
        # TODO: Normalize to [0,1], shape (3xHxW), norm to 
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        x /= 255.0

        with torch.no_grad():
            return self.model.forward(x)


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
        depth_e = torch.mean(outputs[:,0] - target[:,0]) ** 2
        semantic_e = torch.mean(outputs[:,1:] - target[:,1:] / self.num_cats) ** 2
        loss = depth_e + semantic_e
        return loss

class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 320x320
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=42+1+1, # Semantic + depth
                out_channels=48,
                kernel_size=5,
                stride=3,
            ), # b, 128, 106
            nn.ReLU(),
            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=5,
                stride=3,
                padding=2
            ), # b, 192, 36
            nn.ReLU(),
            nn.Conv2d(
                in_channels=96,
                out_channels=128,
                kernel_size=5,
                stride=3,
                padding=1
            ), # b, 256, 12
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=160,
                kernel_size=5,
                stride=3,
                padding=1
            ), # b, 256, 4
            nn.ReLU(), # Final shape 1024 after flattening
        )


    def forward(self, x):
        return self.encoder(x)

class CNNDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(160, 128, kernel_size=5, stride=3, padding=1), # b, 192, 14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 96, kernel_size=5, stride=3, padding=1), # b, 192, 44
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=5, stride=3, padding=2), # b, 192, 134
            nn.ReLU(),
            nn.ConvTranspose2d(48, 43, kernel_size=5, stride=3), # b, 192, 468
            nn.Tanh(),
        )

class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 320x320
        self.encoder = nn.Sequential(
            # Semantic + target + depth
            nn.Conv2d(42+1+1, 64, 5, stride=3), # b, 64, 106
            nn.ReLU(),
            nn.Conv2d(64, 80, 5, stride=3,padding=2), # b, 128, 36
            nn.ReLU(),
            nn.Conv2d(80,96,5,stride=3,padding=1), # b, 160, 12
            nn.ReLU(),
            nn.Conv2d(96,112,5, stride=3,padding=1), # b, 192, 4
            nn.ReLU(), # Final shape 1024 after flattening
            nn.Conv2d(112,128,3,stride=1), # b, 192, 4
            nn.ReLU(), # Final shape 1024 after flattening
        )


    def forward(self, x):
        '''
        for layer in self.encoder:
            x = layer(x)
            print(x.shape)
        return x
        '''
        return self.encoder(x)

class CNNDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 112, kernel_size=3, stride=1), # b, 192, 14
            nn.ReLU(),
            nn.ConvTranspose2d(112, 96, kernel_size=5, stride=3, padding=1), # b, 192, 44
            nn.ReLU(),
            nn.ConvTranspose2d(96, 80, kernel_size=5, stride=3, padding=1), # b, 192, 134
            nn.ReLU(),
            nn.ConvTranspose2d(80, 64, kernel_size=5, stride=3, padding=2), # b, 192, 468
            nn.ReLU(),
            nn.ConvTranspose2d(64, 42+1+1, kernel_size=5, stride=3), # b, 192, 468
            nn.Tanh(),
        )

    def forward(self, x):
        '''
        for layer in self.decoder:
            x = layer(x)
            print(x.shape)
        return x
        '''
        return self.decoder(x)
