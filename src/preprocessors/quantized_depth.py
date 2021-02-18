import numpy as np
import torch
import cv2
from gym import spaces
from habitat_baselines.common.obs_transformers import ObservationTransformer


class QuantizedDepth(ObservationTransformer):
    def __init__(self, env, height_fac=32, width_fac=32):
        self.env = env
        self.dtype = np.float32
        self.facs = np.array((height_fac, width_fac), dtype=np.int32)
        super().__init__()

    def transform_observation_space(self, obs_space):
        if "depth" not in obs_space.spaces:
            return obs_space

        self.sensor_shape = np.array(
            (
                self.env.hab_cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                self.env.hab_cfg.SIMULATOR.DEPTH_SENSOR.WIDTH,
            ),
            dtype=np.int32,
        )
        self.shape = self.sensor_shape // self.facs
        assert np.isclose(
            self.facs * self.shape, self.sensor_shape
        ).all(), "Shapes do not align, change the factors"

        obs_space.spaces["depth"] = spaces.Box(
            # 1 is far, 0 is near
            low=0,
            high=1,
            shape=self.shape,
            dtype=self.dtype,
        )
        return obs_space

    def forward(self, obs):
        if "depth" not in obs:
            return obs

        depth = np.reshape(obs["depth"], (1, *self.sensor_shape))
        # Rather than mean, we care about the closest object per pixel
        # i.e. min_pool
        quant_depth = (
            torch.nn.functional.max_pool2d(torch.FloatTensor(depth), self.facs.tolist())
            .squeeze()
            .numpy()
        )
        obs["depth"] = quant_depth
        self.quant_depth = quant_depth
        return obs

    def visualize(self):
        return (
            254
            * cv2.resize(self.quant_depth, tuple(self.sensor_shape), cv2.INTER_NEAREST)
        ).astype(np.uint8)

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
