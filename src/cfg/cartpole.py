import os
from typing import Dict, Any

from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.a3c import A3CTrainer, A2CTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env, grid_search

from rewards.basic import BasicReward
from rewards.path import PathReward

from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole

from models.ray_graph import RayObsGraph
from models.ray_dnc import DNCMemory
from models.edge_selectors.temporal import TemporalBackedge

from copy import deepcopy
import torch_geometric
import torch


register_env(StatelessCartPole.__name__, StatelessCartPole)
cfg_dir = os.path.abspath(os.path.dirname(__file__))

hiddens = [32, 16, 8]
seq_len = 20
horizon = 200
gsize = 10


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
                "nr_cells": gsize,
                "cell_size": hidden,
                "preprocessor_input_size": hidden,
                "preprocessor_output_size": hidden,
                "preprocessor": torch.nn.Sequential(
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.Tanh(),
                ),
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
            "edge_selectors": TemporalBackedge([1, 2]),
            # "use_prev_action": True,
        },
        "max_seq_len": seq_len,
    }
    graph_models.append(temporal_model)

models = [
    *graph_models,
    rnn_model,
    attn_model,
    no_mem,
    dnc_model,
]


CFG = {
    # Our specific trainer type
    "ray_trainer": PPOTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        # These are rllib/ray specific
        "env_config": {},
        "framework": "torch",
        "model": grid_search(models),
        "num_workers": 2,
        "num_cpus_per_worker": 2,
        "num_gpus": 0.2,
        "env": StatelessCartPole,
        "vf_loss_coeff": 1e-5,
        "horizon": horizon,
        # "lr": 0.0005,
        # "train_batch_size": 2000,
        # "num_sgd_iter": 15,
        # "sgd_minibatch_size": 100,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 1e6},
        "num_samples": 3,
    },
}
