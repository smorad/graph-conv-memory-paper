import numpy as np
import gym
from habitat_baselines.common.obs_transformers import ObservationTransformer


class CompassComponents(ObservationTransformer):
    def __init__(self, env):
        super().__init__()

    def transform_observation_space(self, obs_space):
        if "compass" not in obs_space.spaces:
            return obs_space

        obs_space.spaces["compass"] = gym.spaces.Box(
            high=1.0,
            low=-1.0,
            dtype=np.float,
            shape=(2, 1),
        )

        return obs_space

    def forward(self, obs):
        if "compass" not in obs:
            return obs

        # Correct for over/underflow
        obs["compass"] = np.array([np.sin(obs["compass"]), np.cos(obs["compass"])])
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
