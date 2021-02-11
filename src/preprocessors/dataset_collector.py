import torch
import cv2
import numpy as np
from habitat_baselines.common.obs_transformers import ObservationTransformer

from models.vae import VAE
from semantic_colors import COLORS64


class DataCollector(ObservationTransformer):
    def __init__(self, env, num_cats=43, batch_size=8192, num_batches=16):
        super().__init__()
        self.dtype = np.float32
        self.env = env
        self.fwd_ctr = 0
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.storage = np.zeros((batch_size, 43 + 1, 32, 32))

    def transform_observation_space(self, obs_space):
        assert (
            "semantic" in obs_space.spaces and "depth" in obs_space.spaces
        ), "VAE requires depth and semantic images"
        # Pop these so rllib doesn't spend forever training with them
        obs_space.spaces.pop("semantic", None)
        obs_space.spaces.pop("depth", None)
        return obs_space

    def forward(self, obs):
        self.in_sem = obs.pop("semantic", None)
        self.in_depth = obs.pop("depth", None)
        img = self.to_img(self.in_sem, self.in_depth)
        self.storage[self.fwd_ctr % self.batch_size] = img
        self.fwd_ctr += 1

        if self.fwd_ctr != 0 and self.fwd_ctr % self.batch_size == 0:
            torch.save(self.storage, f"img_batch_{self.fwd_ctr // self.batch_size}")
            self.storage = np.zeros((self.batch_size, 43 + 1, 32, 32))

        if self.fwd_ctr >= self.batch_size * self.num_batches:
            raise Exception("Done")

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

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
