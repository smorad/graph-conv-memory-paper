from rayenv import NavEnv
from models.ae import CNNEncoder, CNNAutoEncoder
from habitat_baselines.common.obs_transformers import ObservationTransformer
from gym.spaces import Dict, Box
import numpy as np
import torch

# This file containers preprocessors
# that take observations from NavEnv (or children)
# and compress them using a CNN before sending
# them to the rllib policies


class AutoEncoderPP(ObservationTransformer):
    validation_set = None
    loss = -1

    def __init__(self, env, val_set_len=100):
        super().__init__()
        self.nn = CNNAutoEncoder()
        self.val_set_len = val_set_len

    def add_to_val_set(self, x):
        if not self.validation_set:
            self.validation_set = x
        elif self.validation_set.shape < self.val_set_len:
            self.validation_set = torch.cat((self.validation_set, x), 0)
        else:
            pass

    def transform_observation_space(self, obs_space):
        obs_space.spaces.pop("semantic")
        obs_space.spaces.pop("depth")

    def encoder_forward(self, obs):
        semantic = obs.pop("semantic")
        # TODO: convert sem_tgt (str) to int
        # https://github.com/facebookresearch/habitat-lab/blob/b7a93bc493f7fb89e5bf30b40be204ff7b5570d7/habitat/tasks/nav/object_nav_task.py
        # sem_tgt = obs['objectgoal']
        depth = obs.pop("depth")

        x = np.zeros(
            (semantic.shape[0] + 1 + 1, *semantic.shape[1:]), dtype=np.float32
        )  # Sem, sem_tgt, depth
        x[0 : semantic.shape[0]] = semantic
        # x[semantic_shape[0] = sem_tgt
        x[semantic.shape[0] + 1] = np.squeeze(depth)
        net_in = torch.from_numpy(x).unsqueeze(0)

        self.add_to_val_set(net_in)

    def forward(self, obs):
        # We actually want to train here
        # so forward should call nn.forward, compute loss, and backprop

        return obs


class EncoderPP(ObservationTransformer):
    def __init__(self):
        super().__init__()
        self.nn = CNNEncoder()

    def transform_observation_space(self, obs_space):
        if "semantic" not in obs_space.spaces or "depth" not in obs_space.spaces:
            raise Exception("EncoderEnv requires semantic and depth sensors")

        obs_space.spaces.pop("semantic")
        obs_space.spaces.pop("depth")
        obs_space.spaces["encoder"] = Box(
            high=np.finfo(np.float32).max,
            low=np.finfo(np.float32).min,
            shape=(512,),
            dtype=np.float32,
        )

        return obs_space

    def forward(self, obs):
        semantic = obs.pop("semantic")
        # TODO: convert sem_tgt (str) to int
        # https://github.com/facebookresearch/habitat-lab/blob/b7a93bc493f7fb89e5bf30b40be204ff7b5570d7/habitat/tasks/nav/object_nav_task.py
        # sem_tgt = obs['objectgoal']
        depth = obs.pop("depth")

        x = np.zeros(
            (semantic.shape[0] + 1 + 1, *semantic.shape[1:]), dtype=np.float32
        )  # Sem, sem_tgt, depth
        x[0 : semantic.shape[0]] = semantic
        # x[semantic_shape[0] = sem_tgt
        x[semantic.shape[0] + 1] = np.squeeze(depth)
        net_in = torch.from_numpy(x).unsqueeze(0)

        obs["encoder"] = self.nn(net_in).flatten().detach().numpy()
        return obs

    def from_config(cls, config):
        # TODO: Figure out if we need this
        raise NotImplementedError()
