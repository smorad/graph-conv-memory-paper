import numpy as np
from gym import spaces
from habitat_baselines.common.obs_transformers import ObservationTransformer


class SemanticFloat(ObservationTransformer):
    def __init__(self, env, num_cats=43):
        self.env = env
        self.num_cats = num_cats
        self.dtype = np.float32
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

        uniqs, counts = np.unique(obs["semantic"])
        detected_cats = self.env.instance_to_cat[uniqs]
        output = np.zeros(self.shape, dtype=self.dtype)
        for i in range(len(detected_cats)):
            output[detected_cats[i]] += counts[i]
        output /= output.sum()
        obs["semantic"] = output
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
