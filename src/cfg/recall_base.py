import os
from ray.tune import register_env
from recall_env import RecallEnv
from ray.rllib.agents.impala import ImpalaTrainer
from models.ray_graph import RayObsGraph


register_env(RecallEnv.__name__, RecallEnv)


model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": 16,
        "gcn_output_size": 16,
        "gcn_hidden_size": 16,
        "gcn_num_layers": 3,
        "edge_selectors": ["learned"],
    },
    "max_seq_len": 8,
}

CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        "env_config": {"dim": 4, "max_items": 4, "max_queries": 4},
        # These are rllib/ray specific
        "framework": "torch",
        "model": model,
        "num_workers": 1,
        # Total GPU usage: num_gpus (trainer proc) + num_gpus_per_worker (workers)
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
        # this corresponds to the number of learner GPUs used,
        # not the total used for the environments/rollouts
        "num_gpus": 1,
        # Size of batches (in timesteps) placed in the learner queue
        "rollout_fragment_length": 16,
        # Total number of timesteps to train per batch
        "train_batch_size": 64,
        # Vtrace seems to use a ton of GPU memory, and also appears to leak it when using
        # graph models
        # In the paper, it does not seem to make a huge performance difference
        # unless experience replay is used
        # "vtrace": False,
        "lr": 0.0001,
        "env": RecallEnv.__name__,
        # For evaluation
        # How many epochs/train iters
        # "evaluation_interval": 10,
        # "evaluation_num_episodes": 10,
        #
        # "evaluation_config": val_env_cfg,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    },
    # Env to be loaded when mode == human
    "human_env": RecallEnv,
}

if os.environ.get("DEBUG", False):
    print("-------DEBUG MODE---------")
    # CFG['ray']['num_envs_per_worker'] = 1
    CFG["ray"]["num_workers"] = 0
    # CFG['ray']['train_batch_size'] = 64
    # CFG['ray']['rollout_fragment_length'] = 32
