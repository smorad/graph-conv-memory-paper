from habitat_baselines.common.obs_transformers import ObservationTransformer


class GhostRGB(ObservationTransformer):
    def __init__(self):
        super().__init__()

    def transform_observation_space(self, obs_space):
        obs_space.spaces.pop("rgb", None)
        return obs_space

    def forward(self, obs):
        obs.pop("rgb", None)
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
