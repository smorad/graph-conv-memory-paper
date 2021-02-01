import habitat
import numpy as np
from gym import spaces
import habitat_sim
from habitat_baselines.common.obs_transformers import ObservationTransformer
from typing import List, Any, Union, Optional, cast, Dict


class SemanticMask(ObservationTransformer):
    def __init__(self, env, num_cats=42):
        self.num_cats = 42
        self.env = env
        super().__init__()

    def transform_observation_space(self, obs_space):
        if "semantic" not in obs_space.spaces:
            return obs_space

        self.shape = (self.num_cats, *obs_space["semantic"].shape)
        obs_space.spaces["semantic"] = spaces.Box(
            low=0, high=1, shape=self.shape, dtype=np.uint32
        )
        return obs_space

    def forward(self, obs):
        if "semantic" not in obs:
            return obs

        layers = np.zeros(self.shape, dtype=np.uint32)
        # 1 channel, pixel == instance_id => n channel, channel == obj_id, pixel == 1
        for inst_id, obj_id in self.env.semantic_label_lookup.items():
            idxs = np.where(obs["semantic"].flat == inst_id)
            layers[obj_id].flat[idxs] = 1
        obs["semantic"] = layers
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
