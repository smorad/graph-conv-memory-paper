from torchvision.models import resnext50_32x4d
from torchvision.models.resnet import ResNet

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=512, z_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(43 + 1, 64, 5, stride=3),  # b, 64, 10
            nn.ReLU(),
            nn.Conv2d(64, 96, 4, stride=2),  # b, 96, 4, 4
            nn.ReLU(),
            nn.Conv2d(96, 128, 2, stride=2),  # b, 128, 2, 2
            nn.ReLU(),
            nn.Flatten(),
        )

        """
        self.encoder = nn.Sequential(
            nn.Conv2d(42 + 1 + 1, 64, 5, stride=3),  # b, 64, 106
            nn.ReLU(),
            nn.Conv2d(64, 80, 5, stride=3, padding=2),  # b, 128, 36
            nn.ReLU(),
            nn.Conv2d(80, 96, 5, stride=3, padding=1),  # b, 160, 12
            nn.ReLU(),
            nn.Conv2d(96, 112, 5, stride=3, padding=1),  # b, 192, 4
            nn.ReLU(),
            nn.Conv2d(112, 128, 3, stride=1),  # b, 192, 4
            nn.ReLU(),  # Final shape (128x2x2) 512 after flattening
            nn.Flatten(),
        )
        """

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (h_dim // 4, 2, 2)),  # TODO: This should be variable
            nn.ConvTranspose2d(128, 96, 2, stride=2),  # b, 128, 2, 2
            nn.ReLU(),
            nn.ConvTranspose2d(96, 64, 4, stride=2),  # b, 96, 4, 4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 43 + 1, 5, stride=3),  # b, 64, 10
            nn.Sigmoid(),
        )
        """
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (h_dim // 4, 2, 2)),  # TODO: this should be variable
            nn.ConvTranspose2d(128, 112, kernel_size=3, stride=1),  # b, 192, 14
            nn.ReLU(),
            nn.ConvTranspose2d(
                112, 96, kernel_size=5, stride=3, padding=1
            ),  # b, 192, 44
            nn.ReLU(),
            nn.ConvTranspose2d(
                96, 80, kernel_size=5, stride=3, padding=1
            ),  # b, 192, 134
            nn.ReLU(),
            nn.ConvTranspose2d(
                80, 64, kernel_size=5, stride=3, padding=2
            ),  # b, 192, 468
            nn.ReLU(),
            nn.ConvTranspose2d(64, 42 + 1 + 1, kernel_size=5, stride=3),  # b, 192, 468
            # nn.Sigmoid(),
        )
        """

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(logvar.device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
        # BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD
