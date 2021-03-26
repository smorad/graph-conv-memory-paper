import torch
import gym
import cv2
import numpy as np
from habitat_baselines.common.obs_transformers import ObservationTransformer

from models.depth_vae import DepthVAE

# TODO: This file needs to be cleaned up


class PPDepthVAE(ObservationTransformer):
    def __init__(self, env, cpt_path="/root/vnav/depth_vae.pt"):
        super().__init__()
        self.dtype = np.float32
        self.env = env
        self.net = torch.load(cpt_path).to("cpu")
        # self.net.load_state_dict(cpt['model_state_dict'])
        self.net.eval()

    def transform_observation_space(self, obs_space):
        assert (
            "rgb" in obs_space.spaces and "depth" in obs_space.spaces
        ), "VAE requires rgb and depth images"
        self.sensor_shape = np.array(
            (
                self.env.hab_cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                self.env.hab_cfg.SIMULATOR.DEPTH_SENSOR.WIDTH,
            ),
            dtype=np.int32,
        )
        self.shape = (self.net.z_dim,)
        obs_space.spaces.pop("rgb", None)
        obs_space.spaces.pop("depth", None)
        obs_space.spaces["vae"] = gym.spaces.Box(
            shape=self.shape,
            dtype=self.dtype,
            high=np.finfo(self.dtype).max,
            low=np.finfo(self.dtype).min,
        )
        return obs_space

    def forward(self, obs):
        obs.pop("rgb", None)
        self.in_depth = obs.pop("depth", None)
        with torch.no_grad():
            img = self.to_img(self.in_depth)
            self.z, self.mu, self.logvar = self.net.encode(img)
            obs["vae"] = self.z.squeeze().numpy()
        return obs

    def reconstruct(self):
        recon = self.net.decode(self.z)
        out_depth = recon[0, 0, :, :].detach().numpy()
        return out_depth

    def to_img(self, depth):
        """Build obs into an image tensor for feeding to nn"""
        dep = torch.from_numpy(depth)

        # result: [h, w, 4]
        x = dep.permute(2, 0, 1)
        assert x.max() <= 1.0 and x.min() >= 0.0
        # [batch, channel, cols, rows]
        return x.unsqueeze(0)

    def visualize(self, scale=4):
        # TODO: Clean this up
        out_depth = self.reconstruct()
        depth_out = (
            cv2.resize(
                255 * out_depth, tuple(scale * self.sensor_shape), cv2.INTER_NEAREST
            )
        ).astype(np.uint8)
        depth_out = np.stack((depth_out,) * 3, axis=-1)

        return depth_out

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
