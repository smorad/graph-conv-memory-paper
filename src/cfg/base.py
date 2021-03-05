import os

from ray.rllib.agents.impala import ImpalaTrainer
from ray.tune import register_env

from preprocessors.compass_fix import CompassFix
from preprocessors.semantic.quantized_mesh import QuantizedSemanticMask
from preprocessors.quantized_depth import QuantizedDepth
from preprocessors.ghost_rgb import GhostRGB
from preprocessors.autoencoder.ppae import PPAE

from rewards.basic import BasicReward
from rewards.path import PathReward

from rayenv import NavEnv
from custom_metrics import CustomMetrics


register_env(NavEnv.__name__, NavEnv)
cfg_dir = os.path.abspath(os.path.dirname(__file__))

CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        "env_config": {
            # Path to the habitat yaml config, that specifies sensor info,
            # which maps to use, etc.
            "hab_cfg_path": f"{cfg_dir}/objectnav_mp3d_train_val_mini.yaml",
            # Habitat preprocessors change the observation space in the simulator
            # These are loaded and run in-order
            "preprocessors": {
                "compass": CompassFix,
                "semantic": QuantizedSemanticMask,
                "depth": QuantizedDepth,
                "rgb_visualization": GhostRGB,
                "semantic_and_depth_autoencoder": PPAE,
            },
            # Multiple reward functions may be implemented at once,
            # they are summed together
            "rewards": {"stop_goal": BasicReward, "goal_path": PathReward},
        },
        # These are rllib/ray specific
        "framework": "torch",
        "model": {"framestack": False},
        "num_workers": 1,
        # Total GPU usage: num_gpus (trainer proc) + num_gpus_per_worker (workers)
        "num_gpus_per_worker": 0.15,
        "num_cpus_per_worker": 2,
        # this corresponds to the number of learner GPUs used,
        # not the total used for the environments/rollouts
        "num_gpus": 0.15,
        # Size of batches (in timesteps) placed in the learner queue
        "rollout_fragment_length": 256,
        # Total number of timesteps to train per batch
        "train_batch_size": 1024,
        "lr": 0.0001,
        "env": NavEnv.__name__,
        "callbacks": CustomMetrics,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    },
}
