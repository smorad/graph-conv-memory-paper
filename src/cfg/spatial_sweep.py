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
hidden = 32
gsize = seq_len + 1
act_dim = 3


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

dgc3 = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
dgc3.name = "GraphConv_3h"

gcn = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGCNConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGCNConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
gcn.name = "GCN_2h"

gcn3 = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGCNConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGCNConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGCNConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
gcn3.name = "GCN_3h"

sage = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseSAGEConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseSAGEConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
sage.name = "SAGE_2h"

sage3 = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseSAGEConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseSAGEConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseSAGEConv(hidden, hidden), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
sage3.name = "SAGE_3h"

gin_nn = torch.nn.Sequential(
    torch.nn.Linear(hidden, hidden),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden, hidden),
    torch.nn.Tanh(),
)
gin = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGINConv(gin_nn, train_eps=True), "x, adj -> x"),
        (torch_geometric.nn.DenseGINConv(gin_nn, train_eps=True), "x, adj -> x"),
    ],
)
gin.name = "GIN_2h_tanh"

gnns = [
    gcn,
    gcn3,
    sage,
    sage3,
    gin,
    dgc,
    dgc3,
]

for gnn in gnns:
    # Monkey patch reprs so tensorboard can parse
    # logfile names
    gnn.__class__.__repr__ = lambda self: self.name

models = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gnn_input_size": hidden,
        "gnn_output_size": hidden,
        "gnn": grid_search(gnns),
        "use_prev_action": True,
        "edge_selectors": SpatialEdge(max_distance=0.25, pose_slice=slice(2, 4)),
    },
    "max_seq_len": seq_len,
}


CFG = base.CFG
CFG["ray"]["num_workers"] = 8
CFG["ray"]["model"] = models  # grid_search(models)

# this corresponds to the number of learner GPUs used,
# not the total used for the environments/rollouts
# Since this is the bottleneck, we let it use an entire 1024
CFG["ray"]["num_gpus"] = 1.0

# For rollout workers
CFG["ray"]["num_gpus_per_worker"] = 0.2
CFG["ray"]["num_cpus_per_worker"] = 2

# At batch sizes of 1024 and 2048, GPU learn time is roughly the same per sample
CFG["ray"]["train_batch_size"] = 1024
CFG["ray"]["rollout_fragment_length"] = seq_len

CFG["tune"] = {
    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    "stop": {"info/num_steps_trained": 10e6},
}


if os.environ.get("DEBUG", False):
    # CFG["ray"]["model"] = bernoulli_reg
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
