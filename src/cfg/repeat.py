from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.tune import register_env
from ray.tune import grid_search
from models.edge_selectors.temporal import TemporalBackedge
from models.ray_graph import RayObsGraph
import torch_geometric
import torch
from copy import deepcopy
from typing import Dict, Any
from models.ray_dnc import DNCMemory

from cfg import base

register_env(RepeatAfterMeEnv.__name__, RepeatAfterMeEnv)
seq_len = 20
hidden = 32
gsize = seq_len + 1

# These are specific to our habitat-based environment
env_cfg = {"delay": 1, "episode_len": seq_len}

dgc = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
dgc.name = "GraphConv_1h"
base_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gnn_input_size": hidden,
        "gnn_output_size": hidden,
        "gnn": dgc,
        # "use_prev_action": True,
    },
    "max_seq_len": seq_len,
}
temporal_model = deepcopy(base_model)
temporal_model["custom_model_config"]["edge_selectors"] = TemporalBackedge()
no_mem: Dict[str, Any] = {
    "fcnet_hiddens": [hidden],
    "fcnet_activation": "tanh",
}
rnn_model = {
    **no_mem,
    "use_lstm": True,
    "max_seq_len": seq_len,
    "lstm_cell_size": hidden,
    # "lstm_use_prev_action": True,
}  # type: ignore
dnc = {
    "custom_model": DNCMemory,
    "custom_model_config": {
        "hidden_size": hidden,
        # "num_layers": 1,
        # "num_hidden_layers": 1,
        "read_heads": 2,
        "nr_cells": 4,
        "cell_size": 4,
    },
    "max_seq_len": seq_len,
}

models = [
    # temporal_model,
    # rnn_model,
    dnc,
]

CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        # These are rllib/ray specific
        "env_config": {},
        "framework": "torch",
        "model": grid_search(models),
        "num_workers": 2,
        # "num_cpus_per_worker": 16,
        # "num_envs_per_worker": 8,
        # "num_gpus_per_worker": 0.1,
        "num_gpus": 1,
        "env": RepeatAfterMeEnv.__name__,
        "entropy_coeff": 0.001,
        "vf_loss_coeff": 1e-4,
        "gamma": 0.99,
        # Use these to prevent losing reward on the final step
        # due to overflow
        # "horizon": seq_len - 1,
        # "batch_mode": "complete_episodes",
        "vtrace": False,
        "lr": 0.001,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 1e6},
    },
}
"""
CFG = base.CFG
CFG["ray"]["env_config"] = env_cfg
CFG["ray"]["env"] = RepeatAfterMeEnv.__name__
CFG["ray_trainer"] = PPOTrainer
CFG["ray"]["num_gpus"] = 0.5
CFG["ray"]["num_gpus_per_worker"] = 0
CFG["ray"]["num_envs_per_worker"] = 8
CFG["ray"].pop("lr", None)
CFG["ray"]["gamma"] = 0.9
CFG["ray"]["vf_loss_coeff"] = 1e-5
CFG["ray"]["num_sgd_iter"] = 5
CFG["ray"].pop("replay_proportion", None)
CFG["ray"].pop("replay_buffer_num_slots", None)
CFG["ray"].pop("callbacks", None)
CFG["ray"].pop("evaluation_config", None)
CFG["ray"].pop("human_env", None)
"""
