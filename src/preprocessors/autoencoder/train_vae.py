import torch
import cv2
import numpy as np
from habitat_baselines.common.obs_transformers import ObservationTransformer

from models.vae import VAE
from semantic_colors import COLORS64

# TODO: This file needs to be cleaned up


class VAETrainer(ObservationTransformer):
    def __init__(self, env, num_cats=43, height_fac=10, width_fac=10):
        super().__init__()
        self.dtype = np.float32
        self.env = env
        self.nn = VAE()
        learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=learning_rate)
        self.losses = []

    def transform_observation_space(self, obs_space):
        assert (
            "semantic" in obs_space.spaces and "depth" in obs_space.spaces
        ), "VAE requires depth and semantic images"
        self.sensor_shape = np.array(
            (
                self.env.hab_cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                self.env.hab_cfg.SIMULATOR.DEPTH_SENSOR.WIDTH,
            ),
            dtype=np.int32,
        )
        # Pop these so rllib doesn't spend forever training with them
        obs_space.spaces.pop("semantic", None)
        obs_space.spaces.pop("depth", None)
        return obs_space

    def backprop(self, in_sem, in_depth, out_sem, out_depth, depth_to_sem_ratio=1 / 4):
        sem_factor = 1 / in_sem.shape[1]
        MSE_sem = (
            torch.nn.functional.mse_loss(in_sem, out_sem)
            * (1 / depth_to_sem_ratio)
            * sem_factor
        )
        MSE_depth = torch.nn.functional.mse_loss(in_depth, out_depth)
        KLD = -0.5 * torch.mean(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        loss = MSE_sem + MSE_depth + KLD
        self.losses.append(loss.item())
        if len(self.losses) == 100:
            print("Mean loss:", np.mean(self.losses))
            self.losses.clear()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, obs):
        self.in_sem = obs.pop("semantic", None)
        self.in_depth = obs.pop("depth", None)
        img = self.to_img(self.in_sem, self.in_depth)
        recon, self.mu, self.logvar = self.nn(img)
        self.out_sem = recon[0, :-1, :, :]
        self.out_depth = recon[0, -1, :, :]
        self.optimizer.zero_grad()
        self.backprop(
            torch.Tensor(self.in_sem),
            torch.Tensor(self.in_depth),
            self.out_sem,
            self.out_depth,
        )
        return obs

    def to_img(self, semantic, depth):
        """Build obs into an image tensor for feeding to nn"""
        # [batch, channel, cols, rows]
        channels, cols, rows = (  # type: ignore
            semantic.shape[0] + 1,
            *semantic.shape[1:],  # type : ignore
        )

        depth_channel = semantic.shape[0]
        x = torch.zeros(
            (channels, cols, rows),
            dtype=torch.float32,
        )
        x[0:depth_channel] = torch.Tensor(semantic)
        x[depth_channel] = torch.Tensor(depth).squeeze()
        return x.unsqueeze(0)

    def visualize(self):
        # TODO: Clean this up
        out_depth = self.out_depth.detach().numpy()
        out_sem = self.out_sem.detach().numpy()
        img = np.zeros((*out_depth.shape, 3), dtype=np.uint8)
        for layer_idx in range(out_sem.shape[0]):
            layer = out_sem[layer_idx, :, :]
            filled_px = np.argwhere(np.around(layer) == 1)
            img[filled_px[:, 0], filled_px[:, 1]] = COLORS64[layer_idx]
        sem_out = cv2.resize(
            img, tuple(self.sensor_shape), interpolation=cv2.INTER_NEAREST
        )

        depth_out = (
            255 * cv2.resize(out_depth, tuple(self.sensor_shape), cv2.INTER_NEAREST)
        ).astype(np.uint8)
        depth_out = np.stack((depth_out,) * 3, axis=-1)

        return cv2.hconcat([sem_out, depth_out])

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
