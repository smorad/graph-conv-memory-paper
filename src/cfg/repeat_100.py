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
seq_len = 101
hidden = 32
gsize = seq_len + 1
delay = 20

# These are specific to our habitat-based environment
env_cfg = {
    "repeat_delay": delay,
}

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
temporal_model["custom_model_config"]["edge_selectors"] = TemporalBackedge(hops=[delay])
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
        "nr_cells": gsize,
        "cell_size": hidden,
    },
    "max_seq_len": seq_len,
}
attn_model = {
    **no_mem,
    "use_attention": True,
    "attention_num_transformer_units": 1,
    "attention_dim": hidden,
    "attention_num_heads": 1,
    "attention_head_dim": hidden,
    "attention_position_wise_mlp_dim": hidden,
    "attention_memory_inference": seq_len,
    "attention_memory_training": seq_len,
    # "attention_use_n_prev_actions": 1,
}

models = [
    temporal_model,
    rnn_model,
    attn_model,
    # dnc,
]

CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        # These are rllib/ray specific
        "env_config": env_cfg,
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
        "batch_mode": "complete_episodes",
        "vtrace": False,
        "lr": 0.0005,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 1e6},
    },
}
