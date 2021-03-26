import os

from ray.rllib.agents.impala import ImpalaTrainer
from ray.tune import register_env, grid_search

from preprocessors.compass_fix import CompassFix
from preprocessors.semantic.quantized_mesh import QuantizedSemanticMask
from preprocessors.quantized_depth import QuantizedDepth
from preprocessors.ghost_rgb import GhostRGB

from models.ray_vae_d import DepthRayVAE

from rewards.basic import BasicReward
from rewards.path import PathReward

from rayenv import NavEnv
from custom_metrics import VAEMetrics


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
            "hab_cfg_path": f"{cfg_dir}/objectnav_mp3d_val_vae_d.yaml",
            # Habitat preprocessors change the observation space in the simulator
            # These are loaded and run in-order
            "preprocessors": {
                "compass": CompassFix,
            },
            "rewards": {},
        },
        # These are rllib/ray specific
        "framework": "torch",
        "model": {
            "framestack": False,
            "custom_model": DepthRayVAE,
            "custom_model_config": {
                "z_dim": 64,  # grid_search([64, 128]),
                "depth_weight": 1.0,
                "rgb_weight": 1.0,
                "elbo_beta": 0.1,
            },
        },
        "num_workers": 12,
        "num_cpus_per_worker": 2,
        # Total GPU usage: num_gpus (trainer proc) + num_gpus_per_worker (workers)
        "num_gpus_per_worker": 0.15,
        # this corresponds to the number of learner GPUs used,
        # not the total used for the environments/rollouts
        "num_gpus": 1,
        # Size of batches (in timesteps) placed in the learner queue
        "rollout_fragment_length": 256,
        # Total number of timesteps to train per batch
        "train_batch_size": 1024,
        "lr": 0.01,
        "env": NavEnv,
        "callbacks": VAEMetrics,
    },
    "tune": {
        "goal_metric": {"metric": "custom_metrics/ae_combined_loss", "mode": "min"},
        "stop": {"info/num_steps_trained": 10e6},
    },
}

if os.environ.get("DEBUG", False):
    print("-------DEBUG---------")
    CFG["ray"]["num_workers"] = 1
    CFG["ray"]["model"]["custom_model_config"]["z_dim"] = 64
    print(CFG)
