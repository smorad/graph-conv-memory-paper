from cfg import base
from custom_metrics import EvalMetrics
from models.sparse_ray_graph import RaySparseObsGraph
from ray.tune import grid_search
import torch
import torch_geometric
from models.sparse_edge_selectors.temporal import TemporalEdge
from models.sparse_edge_selectors.distance import SpatialEdge

import os
from copy import deepcopy

# seq_len must be the same as episode length
seq_len = 128
hidden = 64
act_dim = 3

gc = torch_geometric.nn.Sequential(
    "x, edge_index, weights",
    [
        (torch_geometric.nn.GraphConv(64, 64), "x, edge_index -> x"),
        torch.nn.Tanh(),
        (torch_geometric.nn.GraphConv(64, 64), "x, edge_index -> x"),
        torch.nn.Tanh(),
    ],
)

base_graph = {
    "custom_model": RaySparseObsGraph,
    "custom_model_config": {
        "gnn_input_size": hidden,
        "gnn_output_size": hidden,
        "gnn": gc,
        "edge_selectors": None,
        "use_prev_action": True,
    },
    "max_seq_len": seq_len,
}

temp = deepcopy(base_graph)
temp["custom_model_config"]["edge_selectors"] = TemporalEdge(1)

pose = deepcopy(base_graph)
pose["custom_model_config"]["edge_selectors"] = SpatialEdge(
    max_distance=0.25, pose_slice=slice(2, 4)
)

models = [
    temp,
    # pose
]

CFG = base.CFG
CFG["ray"]["num_workers"] = 5
CFG["ray"]["num_gpus"] = 1.0
# For rollout workers
CFG["ray"]["num_gpus_per_worker"] = 0.2
CFG["ray"]["num_cpus_per_worker"] = 2
CFG["ray"]["train_batch_size"] = 1024
CFG["ray"]["rollout_fragment_length"] = seq_len
CFG["tune"] = {
    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    "stop": {"info/num_steps_trained": 5e6},
}

CFG["ray"]["model"] = grid_search(models)

if os.environ.get("DEBUG", False):
    CFG["ray"]["model"] = temp
    CFG["ray"]["num_workers"] = 0
    CFG["ray"]["num_gpus"] = 0.3
    # CFG["ray"]["evaluation_num_workers"] = 1
    # CFG["ray"]["evaluation_interval"] = 1
    # CFG["ray"]["callbacks"] = EvalMetrics
    # CFG["ray"]["num_gpus"] = 0
    # CFG["ray"]["rollout_fragment_length"] = CFG["ray"]["train_batch_size"]
    # CFG["ray"]["model"]["custom_model_config"]["export_gradients"] = True
    CFG["ray"]["train_batch_size"] = 128
    CFG["ray"]["rollout_fragment_length"] = 128
    # CFG["tune"] = {
    #    "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
    #    "stop": {"info/num_steps_trained": 2048},
    # }
