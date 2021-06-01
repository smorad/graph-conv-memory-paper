import os
from typing import Dict, Any

from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A3CTrainer
from ray.tune import register_env, grid_search

from rewards.basic import BasicReward
from rewards.path import PathReward

from memory_env import MemoryEnv
from models.ray_graph import RayObsGraph
from models.ray_dnc import DNCMemory
from models.edge_selectors.temporal import TemporalBackedge
from models.edge_selectors.distance import SpatialEdge

from copy import deepcopy
import torch_geometric
import torch


register_env(MemoryEnv.__name__, MemoryEnv)
cfg_dir = os.path.abspath(os.path.dirname(__file__))

# exp 1
"""
hiddens = [32]
seq_len = 50
num_cards = 8
num_matches = 2
num_unique_cards = num_cards // num_matches
"""
"""
# exp 2
# optimal_solution = 1.75 * 10 = 18
hiddens = [32]
seq_len = 75
num_cards = 10
num_matches = 2
num_unique_cards = num_cards // num_matches
"""
# exp 3
hiddens = [32]
seq_len = 100
num_cards = 12
num_matches = 2
num_unique_cards = num_cards // num_matches

edge_selector = torch_geometric.nn.Sequential(
    "x, adj, weights, N, B",
    [
        # view_flipped discrete
        # obs slices:
        # ({'card': 0, 'flipped_cards': 5, 'flipped_pos': 15, 'pointer_pos': 31},
        # {'card': 5, 'flipped_cards': 15, 'flipped_pos': 31, 'pointer_pos': 39})
        (
            SpatialEdge(
                1e-3,
                # Match current face-up card
                slice(2 * (num_unique_cards + 1), 3 * (num_unique_cards + 1)),
                # Against previous memories of flipped cards
                slice(0, num_unique_cards + 1),
            ),
            "x, adj, weights, N, B -> adj, weights",
        ),
        (TemporalBackedge([1, 2]), "x, adj, weights, N, B -> adj, weights"),
    ],
)
gsize = seq_len + 2


no_mem = [
    {
        "fcnet_hiddens": [hidden, hidden],
        "fcnet_activation": "tanh",
    }
    for hidden in hiddens
]

rnn_model = [
    {
        "fcnet_hiddens": [hidden, hidden],
        "fcnet_activation": "tanh",
        "use_lstm": True,
        "max_seq_len": seq_len,
        "lstm_cell_size": hidden,
        "lstm_use_prev_action": True,
    }
    for hidden in hiddens
]

dnc_model = [
    {
        "custom_model": DNCMemory,
        "custom_model_config": {
            "hidden_size": hidden,
            "nr_cells": hidden,
            "cell_size": hidden,
            "preprocessor_input_size": hidden,
            "preprocessor_output_size": hidden,
            "preprocessor": torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.Tanh(),
            ),
            "use_prev_action": True,
        },
        "max_seq_len": seq_len,
    }
    for hidden in hiddens
]

attn_model = [
    {
        "fcnet_hiddens": [hidden, hidden],
        "fcnet_activation": "tanh",
        "use_attention": True,
        "attention_num_transformer_units": 1,
        "attention_dim": hidden,
        "attention_num_heads": 1,
        "attention_head_dim": hidden,
        "attention_position_wise_mlp_dim": hidden,
        "attention_memory_inference": seq_len,
        "attention_memory_training": seq_len,
        "attention_use_n_prev_actions": seq_len,
    }
    for hidden in hiddens
]


graph_models = []
for hidden in hiddens:

    dgc = torch_geometric.nn.Sequential(
        "x, adj, weights, B, N",
        [
            # Mean and sum aggregation perform roughly the same
            # Preprocessor with 1 layer did not help
            (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
            (torch.nn.Tanh()),
            (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
            (torch.nn.Tanh()),
        ],
    )
    dgc.name = "GraphConv_2h"
    sgm_model = {
        "custom_model": RayObsGraph,
        "custom_model_config": {
            "graph_size": gsize,
            "gnn_input_size": hidden,
            "gnn_output_size": hidden,
            "gnn": dgc,
            "edge_selectors": edge_selector,
            "use_prev_action": True,
        },
        "max_seq_len": seq_len,
    }
    graph_models.append(sgm_model)

models = [
    *graph_models,
    *attn_model,
    *rnn_model,
    *no_mem,
    *dnc_model,
]


CFG = {
    # Our specific trainer type
    "ray_trainer": A3CTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        "env_config": {
            "num_matches": num_matches,
            "num_cards": num_cards,
            "mode": "view_flipped",
            "episode_length": seq_len - 1,
            "discrete": True,
        },
        "framework": "torch",
        "model": grid_search(models),
        "num_envs_per_worker": 4,
        "num_cpus_per_worker": 2,
        "num_gpus": 0.2,
        "env": MemoryEnv.__name__,
        "entropy_coeff": 0.001,
        "vf_loss_coeff": 0.05,
        "lr": 0.0005,
        "train_batch_size": 2000,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 10e6},
        "num_samples": 3,
    },
}


if os.environ.get("DEBUG", False):
    CFG["ray"]["model"] = graph_models[0]
    CFG["ray_trainer"] = ImpalaTrainer
    CFG["ray"]["num_workers"] = 0
    CFG["ray"]["num_gpus"] = 0.25
    CFG["ray"]["train_batch_size"] = 128
    CFG["ray"]["rollout_fragment_length"] = 64
    CFG["tune"] = {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 128},
    }
