from cfg import base
from models.ray_graph import RayObsGraph
from models.ray_dnc import DNCMemory
from models.edge_selectors.temporal import TemporalBackedge
from models.edge_selectors.bernoulli import BernoulliEdge
from ray.tune import grid_search

import os

seq_len = 128
hidden = 64
gsize = seq_len + 1

no_mem = {
    "fcnet_hiddens": [hidden, hidden],
    "fcnet_activation": "tanh",
}

base_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_output_size": hidden,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 1,
        "edge_selectors": [],
    },
    "max_seq_len": seq_len,
}

temporal_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_output_size": hidden,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 1,
        "edge_selectors": [TemporalBackedge],
    },
    "max_seq_len": seq_len,
}

bernoulli_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gcn_output_size": hidden,
        "gcn_hidden_size": hidden,
        "gcn_hidden_layers": 1,
        "edge_selectors": [BernoulliEdge],
    },
    "max_seq_len": seq_len,
}

rnn_model = {"use_lstm": True, "max_seq_len": seq_len, "lstm_cell_size": hidden}

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
    no_mem,
    base_model,
    rnn_model,
    dnc_model,
]  # , temporal_model, bernoulli_model]

CFG = base.CFG
CFG["ray"]["num_workers"] = 12
CFG["ray"]["model"] = base_model  # grid_search(models)

# this corresponds to the number of learner GPUs used,
# not the total used for the environments/rollouts
# Since this is the bottleneck, we let it use an entire 1024
CFG["ray"]["num_gpus"] = 1

# For rollout workers
CFG["ray"]["num_gpus_per_worker"] = 0.16
CFG["ray"]["num_cpus_per_worker"] = 2

# At batch sizes of 1024 and 2048, GPU learn time is roughly the same per sample
CFG["ray"]["train_batch_size"] = 1024
CFG["ray"]["rollout_fragment_length"] = 128

CFG["tune"] = {
    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    "stop": {"info/num_steps_trained": 10e6},
}

if os.environ.get("DEBUG", False):
    CFG["ray"]["model"] = base_model
    # CFG["ray"]["num_gpus"] = 0
    CFG["ray"]["num_workers"] = 0
    # CFG["ray"]["model"]["custom_model_config"]["export_gradients"] = True
    CFG["ray"]["train_batch_size"] = 64
    CFG["ray"]["rollout_fragment_length"] = 32
    CFG["tune"] = {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 2048},
    }
