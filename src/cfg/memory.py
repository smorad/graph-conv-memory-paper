import os
from typing import Dict, Any

from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env, grid_search

from rewards.basic import BasicReward
from rewards.path import PathReward

from memory_env import MemoryEnv
from models.ray_graph import RayObsGraph
from models.ray_dnc import DNCMemory
from models.edge_selectors.temporal import TemporalBackedge
from models.edge_selectors.distance import EuclideanEdge

from copy import deepcopy
import torch_geometric
import torch


register_env(MemoryEnv.__name__, MemoryEnv)
cfg_dir = os.path.abspath(os.path.dirname(__file__))

# hidden = 8
hiddens = [32]
seq_len = 100
gsize = 10  # seq_len + 1


no_mem = grid_search(
    [
        {
            "fcnet_hiddens": [hidden, hidden],
            "fcnet_activation": "tanh",
        }
        for hidden in hiddens
    ]
)

rnn_model = grid_search(
    [
        {
            "fcnet_hiddens": [hidden, hidden],
            "fcnet_activation": "tanh",
            "use_lstm": True,
            "max_seq_len": seq_len,
            "lstm_cell_size": hidden,
            # "lstm_use_prev_action": True,
        }
        for hidden in hiddens
    ]
)  # type: ignore

dnc_model = grid_search(
    [
        {
            "custom_model": DNCMemory,
            "custom_model_config": {
                "hidden_size": hidden,
                "num_layers": 1,
                "num_hidden_layers": 2,
                "read_heads": 4,
                "nr_cells": gsize,
                "cell_size": hidden,
            },
            "max_seq_len": seq_len,
        }
        for hidden in hiddens
    ]
)

attn_model = grid_search(
    [
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
            # "attention_use_n_prev_actions": 1,
        }
        for hidden in hiddens
    ]
)


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
    temporal_model = {
        "custom_model": RayObsGraph,
        "custom_model_config": {
            "graph_size": gsize,
            "gnn_input_size": hidden,
            "gnn_output_size": hidden,
            "gnn": dgc,
            # 2 edges outperforms 1 when actions are known
            # 1 outperforms 2 when actions are not known
            "edge_selectors": EuclideanEdge(1e-7),
            # "use_prev_action": True,
        },
        "max_seq_len": seq_len,
    }
    graph_models.append(temporal_model)

graph_models = grid_search(graph_models)

models = [
    graph_models,
    rnn_model,
    attn_model,
    no_mem,
    # dnc_model,
]


CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        # These are rllib/ray specific
        # Do not touch these, they perform well for
        # lstm and sgm
        "env_config": {
            "num_matches": 2,
            "num_cards": 16,
            "mode": "view_current",
        },
        "framework": "torch",
        "model": grid_search(models),
        "num_workers": 2,
        "num_cpus_per_worker": 2,
        "num_gpus": 1,
        "env": MemoryEnv,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 0.01,
        "gamma": 0.99,
        "horizon": seq_len,
        "lr": 0.001,
        "train_batch_size": 1000,
        # "grad_clip": 10,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 5e6},
    },
}


if os.environ.get("DEBUG", False):
    CFG["ray"]["model"] = no_mem
    CFG["ray"]["num_workers"] = 0
    CFG["ray"]["num_gpus"] = 0.3
    # CFG["ray"]["evaluation_num_workers"] = 1
    # CFG["ray"]["evaluation_interval"] = 1
    # CFG["ray"]["callbacks"] = EvalMetrics
    # CFG["ray"]["num_gpus"] = 0
    # CFG["ray"]["rollout_fragment_length"] = CFG["ray"]["train_batch_size"]
    # CFG["ray"]["model"]["custom_model_config"]["export_gradients"] = True
    CFG["ray"]["train_batch_size"] = 128
    CFG["ray"]["rollout_fragment_length"] = 64
    CFG["tune"] = {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 128},
    }
