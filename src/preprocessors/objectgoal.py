import numpy as np
from gym import spaces
import cv2
from habitat_baselines.common.obs_transformers import ObservationTransformer
from semantic_colors import COLORS64


class ObjectGoalBoolean(ObservationTransformer):
    def __init__(self, env):
        self.env = env
        self.dtype = np.float32
        super().__init__()

    def transform_observation_space(self, obs_space):
        assert (
            "objectgoal" in obs_space.spaces
        ), "This pp requires the objectgoal sensor"
        assert "semantic" in obs_space.spaces, "This pp requires the semantic sensor"

        # TODO: more gracefully handle both sem_onehot and sem_float
        obs_space.spaces["target_in_view"] = spaces.Box(
            low=0, high=1, shape=(1,), dtype=self.dtype
        )
        return obs_space

    def forward(self, obs):
        assert "objectgoal" in obs, "This pp requires objectgoal input"
        assert "semantic" in obs, "This pp requires semantic input"

        self.tgt_cat = obs["objectgoal"][0].copy()
        # IMPORTANT: This must be loaded after SemanticOneHot
        # as we don't know if obs[semantic] is onehot or an image
        # TODO: Detect if SemanticOneHot is in envs.preprocessors and
        # branch based on that
        self.tgt_found = obs["semantic"][self.tgt_cat]
        obs["target_in_view"] = np.array([self.tgt_found], dtype=self.dtype)
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()

    def visualize(self):
        out = 254 * np.ones((128, 128, 3), dtype=np.uint8)
        tgt_label = self.env.cat_to_str[self.tgt_cat]
        txt0 = f"{tgt_label}({self.tgt_cat})"

        if self.tgt_found:
            txt1 = "FOUND"
        else:
            txt1 = "NOT FOUND"

        color = tuple(COLORS64[self.tgt_cat].tolist())
        cv2.putText(out, txt0, (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(out, txt1, (0, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return out
