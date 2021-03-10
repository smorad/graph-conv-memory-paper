import torch
import gym
import cv2
import numpy as np
from habitat_baselines.common.obs_transformers import ObservationTransformer

from models.vae import VAE
from semantic_colors import COLORS64

# TODO: This file needs to be cleaned up


class PPVAE(ObservationTransformer):
    def __init__(self, env, cpt_path="/root/vnav/vae.pt"):
        super().__init__()
        self.dtype = np.float32
        self.env = env
        self.net = torch.load(cpt_path).to("cpu")
        # self.net.load_state_dict(cpt['model_state_dict'])
        self.net.eval()

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
        self.shape = (self.net.z_dim,)
        obs_space.spaces.pop("semantic", None)
        obs_space.spaces.pop("depth", None)
        obs_space.spaces["vae"] = gym.spaces.Box(
            shape=self.shape,
            dtype=self.dtype,
            high=np.finfo(self.dtype).max,
            low=np.finfo(self.dtype).min,
        )
        return obs_space

    def forward(self, obs):
        self.in_sem = obs.pop("semantic", None)
        self.in_depth = obs.pop("depth", None)
        assert (
            self.in_sem.shape[1:] == self.in_depth.shape
        ), "Semantic and depth must have the same shape to use the vae preprocessor"
        with torch.no_grad():
            img = self.to_img(self.in_sem, self.in_depth)
            self.z, self.mu, self.logvar = self.net.encode(img)
            obs["vae"] = self.z.squeeze()
        return obs

    def reconstruct(self):
        recon = self.net.decode(self.z)
        out_sem = recon[0, :-1, :, :].detach().numpy()
        out_depth = recon[0, -1, :, :].detach().numpy()
        return out_depth, out_sem

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
        out_depth, out_sem = self.reconstruct()
        img = np.zeros((*out_depth.shape, 3), dtype=np.uint8)
        # To single channel where px value is semantic class
        max_likelihood_px = out_sem.argmax(axis=0)
        # Convert to RGB
        img = COLORS64[max_likelihood_px.flat].reshape(*max_likelihood_px.shape, 3)
        sem_out = cv2.resize(
            img, tuple(self.sensor_shape), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

        depth_out = (
            255 * cv2.resize(out_depth, tuple(self.sensor_shape), cv2.INTER_NEAREST)
        ).astype(np.uint8)
        depth_out = np.stack((depth_out,) * 3, axis=-1)

        return cv2.hconcat([sem_out, depth_out])

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
