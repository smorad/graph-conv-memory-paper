import torch
from torch import nn


class RGBDVAE(nn.Module):
    # On CPU this takes 2.5ms to encode
    # on GPU, 886 microsecs
    def __init__(self, h_dim=512, z_dim=64):
        assert h_dim in [512]
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=3),  # b, 64, 10
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # b, 96, 4, 4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),  # b, 128, 2, 2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (h_dim // 4, 2, 2)),  # TODO: This should be variable
            nn.ConvTranspose2d(128, 64, 3, stride=1),  # b, 128, 2, 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),  # b, 96, 4, 4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 5, stride=3),  # b, 64, 10
            nn.Sigmoid(),
        )

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

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
        # BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD
