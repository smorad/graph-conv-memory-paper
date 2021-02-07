import numpy as np
from habitat_baselines.common.obs_transformers import ObservationTransformer


class ObjectGoalBoolean(ObservationTransformer):
    def __init__(self, env):
        self.env = env
        super().__init__()

    def transform_observation_space(self, obs_space):
        return obs_space

    def forward(self, obs):
        if "objectgoal" not in obs or "semantic" not in obs:
            return obs

        tgt_cat = self.env.semantic_label_lookup[obs["objectgoal"]]
        # IMPORTANT: This must be loaded after SemanticOneHot
        # as we don't know if obs[semantic] has object IDs or class IDs
        # TODO: Detect if SemanticOneHot is in envs.preprocessors and
        # branch based on that
        present_cats = obs["semantic"]
        obs["objectgoal"][0] = tgt_cat in present_cats
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
