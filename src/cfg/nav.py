from cfg import base
from models.ray_graph import RayObsGraph
from models.ray_dnc import DNCMemory
from models.edge_selectors.temporal import TemporalBackedge
from models.edge_selectors.bernoulli import BernoulliEdge
from models.edge_selectors.distance import CosineEdge
from models.edge_selectors.dense import DenseEdge
from ray.tune import grid_search
import torch
import torch_geometric

import os

seq_len = 128
hidden = 64
gsize = seq_len + 1
act_dim = 3


no_mem = {
    "fcnet_hiddens": [hidden, hidden],
    "fcnet_activation": "tanh",
}

rnn_model = {**no_mem, "use_lstm": True, "max_seq_len": seq_len, "lstm_cell_size": hidden, "lstm_use_prev_action": True}  # type: ignore

base_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 2,
        "edge_selectors": [],
        "gcn_act_type": torch.nn.Tanh,
        "gcn_conv_type": torch_geometric.nn.DenseGraphConv,
    },
    "max_seq_len": seq_len,
}

sparse_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 2,
        "edge_selectors": [TemporalBackedge(num_hops=1)],
        "gcn_conv_type": torch_geometric.nn.GCNConv,
        "sparse": True,
        "gcn_act_type": torch.nn.Tanh,
    },
    "max_seq_len": seq_len,
}

temporal_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 2,
        # TODO: num_hops only works for 1
        "edge_selectors": [TemporalBackedge(num_hops=1)],
        "gcn_act_type": torch.nn.Tanh,
    },
    "max_seq_len": seq_len,
}

bernoulli_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 2,
        "edge_selectors": [BernoulliEdge(input_size=hidden + act_dim)],
        "gcn_act_type": torch.nn.Tanh,
    },
    "max_seq_len": seq_len,
}

bernoulli_reg = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 2,
        "edge_selectors": [BernoulliEdge(input_size=hidden + act_dim)],
        "gcn_act_type": torch.nn.Tanh,
        "regularize": True,
    },
    "max_seq_len": seq_len,
}


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

cos_dist = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 2,
        # TODO: num_hops only works for 1
        "edge_selectors": [CosineEdge(max_distance=0.2)],
        "gcn_act_type": torch.nn.Tanh,
    },
    "max_seq_len": seq_len,
}

dense = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 2,
        # TODO: num_hops only works for 1
        "edge_selectors": [DenseEdge()],
        "gcn_act_type": torch.nn.Tanh,
    },
    "max_seq_len": seq_len,
}

models = [
    bernoulli_reg,
    bernoulli_model,
    temporal_model,
    no_mem,
    base_model,
    rnn_model,
    # dnc_model,
    cos_dist,
    # dense,
]  # , temporal_model, bernoulli_model]

CFG = base.CFG
CFG["ray"]["num_workers"] = 3
CFG["ray"]["model"] = grid_search(models)

# this corresponds to the number of learner GPUs used,
# not the total used for the environments/rollouts
# Since this is the bottleneck, we let it use an entire 1024
CFG["ray"]["num_gpus"] = 0

# For rollout workers
CFG["ray"]["num_gpus_per_worker"] = 0.3
CFG["ray"]["num_cpus_per_worker"] = 2

# At batch sizes of 1024 and 2048, GPU learn time is roughly the same per sample
CFG["ray"]["train_batch_size"] = 1024
CFG["ray"]["rollout_fragment_length"] = seq_len

CFG["tune"] = {
    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    "stop": {"info/num_steps_trained": 5e6},
}

if os.environ.get("DEBUG", False):
    CFG["ray"]["model"] = bernoulli_reg  # base_model#base_model #sparse_model
    CFG["ray"]["num_workers"] = 0
    # CFG["ray"]["num_gpus"] = 0
    # CFG["ray"]["rollout_fragment_length"] = CFG["ray"]["train_batch_size"]
    # CFG["ray"]["model"]["custom_model_config"]["export_gradients"] = True
    CFG["ray"]["train_batch_size"] = 64
    CFG["ray"]["rollout_fragment_length"] = 32
    # CFG["tune"] = {
    #    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    #    "stop": {"info/num_steps_trained": 2048},
    # }
