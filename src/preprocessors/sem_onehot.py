import numpy as np
from gym import spaces
from habitat_baselines.common.obs_transformers import ObservationTransformer


class SemanticOnehot(ObservationTransformer):
    def __init__(self, env, num_cats=42):
        self.env = env
        self.num_cats = num_cats
        self.dtype = np.int32
        super().__init__()

    def transform_observation_space(self, obs_space):
        if "semantic" not in obs_space.spaces:
            return obs_space

        self.shape = (self.num_cats,)
        obs_space.spaces["semantic"] = spaces.Box(
            low=0, high=1, shape=self.shape, dtype=self.dtype
        )
        return obs_space

    def forward(self, obs):
        if "semantic" not in obs:
            return obs

        uniqs = np.unique(obs["semantic"].flat)
        detected_cats = self.env.semantic_label_lookup[uniqs]
        output = np.zeros(self.shape, dtype=self.dtype)
        output[detected_cats] = 1
        obs["semantic"] = output
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
