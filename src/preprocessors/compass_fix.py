import numpy as np
from habitat_baselines.common.obs_transformers import ObservationTransformer


class CompassFix(ObservationTransformer):
    def __init__(self, env):
        super().__init__()

    def transform_observation_space(self, obs_space):
        if "compass" not in obs_space.spaces:
            return obs_space

        sp = obs_space.spaces["compass"]
        self.low = sp.low
        self.high = sp.high

        return obs_space

    def forward(self, obs):
        if "compass" not in obs:
            return obs

        # Correct for over/underflow
        obs["compass"] = np.clip(obs["compass"], self.low, self.high)
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
