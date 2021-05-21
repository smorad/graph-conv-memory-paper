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
from custom_metrics import VAEMetrics, VAEEvalMetrics


register_env(NavEnv.__name__, NavEnv)
cfg_dir = os.path.abspath(os.path.dirname(__file__))

env_cfg = {
    # Path to the habitat yaml config, that specifies sensor info,
    # which maps to use, etc.
    "hab_cfg_path": f"{cfg_dir}/objectnav_mp3d_train_vae.yaml",
    # Habitat preprocessors change the observation space in the simulator
    # These are loaded and run in-order
    "preprocessors": {},
    # Multiple reward functions may be implemented at once,
    # they are summed together
    # "rewards": {"stop_goal": BasicReward, "goal_path": PathReward},
    "rewards": {},
    # We can't fit all the scenes into memory, so use fewer
    # "scene_proportion": 0.5
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
        "model": {
            "framestack": False,
            "custom_model": DepthRayVAE,
            "custom_model_config": {
                "z_dim": 64,  # grid_search([64, 128]),
                "depth_weight": 1.0,
                "rgb_weight": 1.0,
                "elbo_beta": 0.01,
            },
        },
        "num_workers": 5,
        "num_cpus_per_worker": 2,
        # Total GPU usage: num_gpus (trainer proc) + num_gpus_per_worker (workers)
        "num_gpus_per_worker": 0.15,
        # this corresponds to the number of learner GPUs used,
        # not the total used for the environments/rollouts
        "num_gpus": 1.0,
        # Size of batches (in timesteps) placed in the learner queue
        "rollout_fragment_length": 256,
        # Total number of timesteps to train per batch
        "train_batch_size": 1024,
        "replay_proportion": 5.0,
        "replay_buffer_num_slots": 128,
        # "lr": 0.0005,
        "lr_schedule": [[0, 0.001], [250000, 0.0005], [500000, 0.0001]],
        "env": NavEnv,
        "callbacks": VAEMetrics,
        "evaluation_interval": 10,
        "evaluation_num_episodes": 10,
        "evaluation_config": {
            "env_config": val_env_cfg,
            "callbacks": VAEEvalMetrics,
        },
        # "custom_eval_function": vae_eval,
        "evaluation_num_workers": 1,  # Must be >0 to get OpenGL
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
    CFG["ray"]["env_config"]["scene_proportion"] = 0.05
    print(CFG)
