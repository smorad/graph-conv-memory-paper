from copy import deepcopy
from typing import Dict, Any
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
hidden = 64
gsize = seq_len + 1
act_dim = 3


no_mem: Dict[str, Any] = {
    "fcnet_hiddens": [hidden, hidden],
    "fcnet_activation": "tanh",
}

rnn_model = {
    **no_mem,
    "use_lstm": True,
    "max_seq_len": seq_len,
    "lstm_cell_size": hidden,
    "lstm_use_prev_action": True,
}  # type: ignore

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
    "attention_use_n_prev_actions": 1,
}

dgc = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
dgc.name = "GraphConv_2h"

gin_nn = torch.nn.Sequential(
    torch.nn.Linear(hidden, hidden),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(hidden, hidden),
    torch.nn.LeakyReLU(),
)
gin = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGINConv(gin_nn), "x, adj -> x"),
        (torch.nn.LeakyReLU()),
        (torch_geometric.nn.DenseGINConv(gin_nn), "x, adj -> x"),
    ],
)
gin.name = "GIN_2h_leaky"

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

gin_model = deepcopy(base_model)
gin_model["custom_model_config"]["gnn"] = gin

temporal_model = deepcopy(base_model)
temporal_model["custom_model_config"]["edge_selectors"] = TemporalBackedge()

spatial_model = deepcopy(base_model)
spatial_model["custom_model_config"]["edge_selectors"] = SpatialEdge(
    max_distance=0.25, pose_slice=slice(2, 4)
)

bernoulli_model = deepcopy(base_model)
bernoulli_model["custom_model_config"]["edge_selectors"] = BernoulliEdge(71)

bernoulli_gin = deepcopy(base_model)
bernoulli_gin["custom_model_config"]["edge_selectors"] = BernoulliEdge(71)
bernoulli_gin["custom_model_config"]["gnn"] = gin

dnc_model = {
    "custom_model": DNCMemory,
    "custom_model_config": {
        "hidden_size": hidden,
        "num_layers": 1,
        "num_hidden_layers": 1,
        "read_heads": 4,
        "nr_cells": gsize,
        "cell_size": hidden,
    },
    "max_seq_len": seq_len,
}

models = [
    bernoulli_gin,
    gin_model,
    base_model,
    temporal_model,
    spatial_model,
    bernoulli_model,
    rnn_model,
    no_mem,
]

CFG = base.CFG
CFG["ray"]["num_workers"] = 4
CFG["ray"]["model"] = grid_search(models)

# this corresponds to the number of learner GPUs used,
# not the total used for the environments/rollouts
# Since this is the bottleneck, we let it use an entire 1024
CFG["ray"]["num_gpus"] = 1.0

# For rollout workers
CFG["ray"]["num_gpus_per_worker"] = 0.25
CFG["ray"]["num_cpus_per_worker"] = 2

# At batch sizes of 1024 and 2048, GPU learn time is roughly the same per sample
CFG["ray"]["train_batch_size"] = 1024
CFG["ray"]["rollout_fragment_length"] = seq_len

CFG["tune"] = {
    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    "stop": {"info/num_steps_trained": 5e6},
}

if os.environ.get("DEBUG", False):
    CFG["ray"]["model"] = bernoulli_gin
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
    # CFG["tune"] = {
    #    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    #    "stop": {"info/num_steps_trained": 2048},
    # }
