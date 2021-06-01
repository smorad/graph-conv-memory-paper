from copy import deepcopy
from typing import Dict, Any, List
from cfg import base
from custom_metrics import EvalMetrics
from models.ray_graph import RayObsGraph
from models.ray_dnc import DNCMemory
from models.edge_selectors.temporal import TemporalBackedge
from models.edge_selectors.bernoulli import BernoulliEdge
from models.edge_selectors.distance import CosineEdge, SpatialEdge
from models.edge_selectors.dense import DenseEdge
from ray.tune import grid_search
import torch
import torch_geometric

import os

seq_len = 128
hiddens = [32, 16, 8]
gsize = seq_len + 1
act_dim = 3


no_mem: List[Any] = [
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
        "attention_use_n_prev_actions": 1,
    }
    for hidden in hiddens
]

graph_models = []
for hidden in hiddens:
    dgc = torch_geometric.nn.Sequential(
        "x, adj, weights, B, N",
        [
            (
                torch_geometric.nn.DenseGraphConv(hidden, hidden, aggr="mean"),
                "x, adj -> x",
            ),
            (torch.nn.Tanh()),
            (
                torch_geometric.nn.DenseGraphConv(hidden, hidden, aggr="mean"),
                "x, adj -> x",
            ),
            (torch.nn.Tanh()),
        ],
    )
    dgc.name = "GraphConv_2h_mean"
    dgc.__class__.__repr__ = lambda self: self.name
    base_model = {
        "custom_model": RayObsGraph,
        "custom_model_config": {
            "graph_size": gsize,
            "gnn_input_size": hidden,
            "gnn_output_size": hidden,
            "gnn": dgc,
            "use_prev_action": True,
        },
        "max_seq_len": seq_len,
    }
    # graph_models.append(base_model)

    temporal_model = deepcopy(base_model)
    temporal_model["custom_model_config"]["edge_selectors"] = TemporalBackedge()
    # graph_models.append(temporal_model)

    spatial_model = deepcopy(base_model)
    spatial_model["custom_model_config"]["edge_selectors"] = SpatialEdge(
        max_distance=0.25, a_pose_slice=slice(2, 4)
    )
    graph_models.append(spatial_model)

    vae_model = deepcopy(base_model)
    vae_model["custom_model_config"]["edge_selectors"] = SpatialEdge(
        max_distance=0.1, a_pose_slice=slice(3, 67)
    )
    # graph_models.append(vae_model)


dnc_model = [
    {
        "custom_model": DNCMemory,
        "custom_model_config": {
            "hidden_size": hidden,
            "nr_cells": hidden,
            "cell_size": hidden,
            "preprocessor": torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.Tanh(),
            ),
            "preprocessor_input_size": hidden,
            "preprocessor_output_size": hidden,
            "use_prev_action": True,
        },
        "max_seq_len": seq_len,
    }
    for hidden in hiddens
]

models = [*graph_models, *attn_model, *rnn_model, *no_mem, *dnc_model]

CFG = base.CFG
CFG["ray"]["num_workers"] = 4
CFG["ray"]["model"] = grid_search(models)

# this corresponds to the number of learner GPUs used,
# not the total used for the environments/rollouts
# Since this is the bottleneck, we let it use an entire 1024
CFG["ray"]["num_gpus"] = 0.2

# For rollout workers
CFG["ray"]["num_gpus_per_worker"] = 0.2
CFG["ray"]["num_cpus_per_worker"] = 2

# At batch sizes of 1024 and 2048, GPU learn time is roughly the same per sample
CFG["ray"]["train_batch_size"] = 1024
CFG["ray"]["rollout_fragment_length"] = seq_len

CFG["tune"] = {
    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    "stop": {"info/num_steps_trained": 10e6},
    "num_samples": 1,
}

if os.environ.get("DEBUG", False):
    CFG["ray"]["model"] = dnc_model[0]
    CFG["ray"]["num_workers"] = 1
    CFG["ray"]["num_gpus"] = 0.3
    CFG["ray"]["train_batch_size"] = 128
    CFG["ray"]["rollout_fragment_length"] = 64
    CFG["tune"] = {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 128},
        "num_samples": 3,
    }
