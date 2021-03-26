import os

from ray.rllib.agents.impala import ImpalaTrainer
from ray.tune import register_env

from preprocessors.compass_fix import CompassFix
from preprocessors.semantic.quantized_mesh import QuantizedSemanticMask
from preprocessors.quantized_depth import QuantizedDepth
from preprocessors.ghost_rgb import GhostRGB
from preprocessors.autoencoder.ppae import PPAE
from preprocessors.autoencoder.pprgbd_vae import PPRGBDVAE

from rewards.basic import BasicReward
from rewards.path import PathReward
from rewards.explore import ExplorationReward
from rewards.collision import CollisionReward

from rayenv import NavEnv
from custom_metrics import CustomMetrics


register_env(NavEnv.__name__, NavEnv)
cfg_dir = os.path.abspath(os.path.dirname(__file__))

# These are specific to our habitat-based environment
env_cfg = {
    # Path to the habitat yaml config, that specifies sensor info,
    # which maps to use, etc.
    "hab_cfg_path": f"{cfg_dir}/objectnav_mp3d_train_val_mini.yaml",
    # Habitat preprocessors change the observation space in the simulator
    # These are loaded and run in-order
    "preprocessors": {
        "compass": CompassFix,
        # "semantic": QuantizedSemanticMask,
        # "depth": QuantizedDepth,
        # "rgb_visualization": GhostRGB,
        # "semantic_and_depth_autoencoder": PPAE,
        "rgbd_autoencoder": PPRGBDVAE,
    },
    # Multiple reward functions may be implemented at once,
    # they are summed together
    # "rewards": {"stop_goal": BasicReward, "goal_path": PathReward},
    "rewards": {"exploration": ExplorationReward},
}

# Change the path for our validation set
val_env_cfg = {
    **env_cfg,  # type: ignore
    "hab_cfg_path": f"{cfg_dir}/objectnav_mp3d_train_val_mini.yaml",
}


CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        "env_config": env_cfg,
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
        # Vtrace seems to use a ton of GPU memory, and also appears to leak it when using
        # graph models
        # In the paper, it does not seem to make a huge performance difference
        # unless experience replay is used
        # "vtrace": False,
        "lr": 0.0001,
        "entropy_coeff": 0.05,
        "env": NavEnv.__name__,
        "callbacks": CustomMetrics,
        # "placement_strategy": "SPREAD",
        # For evaluation
        # How many epochs/train iters
        # "evaluation_interval": 10,
        # "evaluation_num_episodes": 10,
        # "evaluation_config": val_env_cfg,
        # "evaluation_num_workers": 1, # Must be >0 to get OpenGL
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    },
    # Env to be loaded when mode == human
    "human_env": NavEnv,
}

CFG["ray"]["vtrace"] = False
