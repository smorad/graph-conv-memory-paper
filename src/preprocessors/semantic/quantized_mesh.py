import habitat
import cv2
import numpy as np
import torch
from gym import spaces
import habitat_sim
from habitat_baselines.common.obs_transformers import ObservationTransformer
from typing import List, Any, Union, Optional, cast, Dict
from semantic_colors import COLORS64


class QuantizedSemanticMask(ObservationTransformer):
    """Downsample an incoming observation of [1,m,n] to [43, m / height_fac, n / width_fac]. This first lookups semantic categories from object IDs, then expands each unique semantic value into an image channel, then downsamples the image."""

    def __init__(self, env, num_cats=43, height_fac=32, width_fac=32):
        self.dtype = np.float32
        self.num_cats = num_cats
        self.env = env
        self.facs = np.array((height_fac, width_fac), dtype=np.int32)
        super().__init__()

    def transform_observation_space(self, obs_space):
        if "semantic" not in obs_space.spaces:
            return obs_space

        self.sensor_shape = np.array(
            (
                self.env.hab_cfg.SIMULATOR.SEMANTIC_SENSOR.HEIGHT,
                self.env.hab_cfg.SIMULATOR.SEMANTIC_SENSOR.WIDTH,
            ),
            dtype=np.int32,
        )
        self.shape = (self.num_cats, *(self.sensor_shape // self.facs))

        assert np.isclose(
            self.facs * self.shape[1:], self.sensor_shape
        ).all(), "Shapes do not align, change the factors"

        obs_space.spaces["semantic"] = spaces.Box(
            low=0,
            high=1,
            shape=self.shape,
            dtype=self.dtype,
        )

        return obs_space

    def forward(self, obs):
        if "semantic" not in obs:
            return obs

        # Convert to category
        new_sem = self.env.convert_sem_obs(obs)
        # Save on computation
        unique_cats = np.unique(new_sem)
        out_img = np.zeros(self.shape)
        # For each unique category, build a downsized layer
        # Do this at the same time to reduce memory usage
        """
        full = torch.Tensor(new_sem == unique_cats)
        downsampled = torch.nn.function.max_pool2d(full)
        out_img = downsampled.squeeze().numpy()
        """

        tmp = np.zeros((self.num_cats, *self.sensor_shape))
        for cat in unique_cats:
            tmp[cat, :, :] = torch.Tensor(new_sem == cat)

        downsized = torch.nn.functional.max_pool2d(
            torch.Tensor(tmp), self.facs.tolist()
        )
        out_img = downsized.squeeze().numpy()
        # out_img[cat,:,:] = downsized_layer.squeeze().numpy()

        obs["semantic"] = out_img
        self.downsized_sem = out_img

        return obs

    def visualize(self):
        img = np.zeros((*self.shape[1:], 3), dtype=np.uint8)
        for layer_idx in range(self.downsized_sem.shape[0]):
            layer = self.downsized_sem[layer_idx, :, :]
            filled_px = np.argwhere(layer == 1)
            img[filled_px[:, 0], filled_px[:, 1]] = COLORS64[layer_idx]
        return cv2.resize(
            img, tuple(self.sensor_shape), interpolation=cv2.INTER_NEAREST
        )

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
