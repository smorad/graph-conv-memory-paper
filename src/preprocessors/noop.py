from habitat_baselines.common.obs_transformers import ObservationTransformer


class NoopPP(ObservationTransformer):
    def __init__(self, env):
        super().__init__()

    def transform_observation_space(self, obs_space):
        return obs_space

    def forward(self, obs):
        return obs

    def from_config(self, cfg):
        raise NotImplementedError()
