from cfg import base
from custom_metrics import EvalMetrics
from models.sparse_ray_graph import RaySparseObsGraph
from ray.tune import grid_search
import torch
import torch_geometric
from models.sparse_edge_selectors.temporal import TemporalEdge

import os

# seq_len must be the same as episode length
seq_len = 128
hidden = 64
act_dim = 3

base_graph = {
    "custom_model": RaySparseObsGraph,
    "custom_model_config": {
        "gnn_input_size": hidden,
        "gnn_output_size": hidden,
        "gnn": torch_geometric.nn.Sequential(
            "x, edge_index, weights",
            [
                (torch_geometric.nn.GraphConv(64, 64), "x, edge_index -> x"),
                torch.nn.Tanh(),
                (torch_geometric.nn.GraphConv(64, 64), "x, edge_index -> x"),
                torch.nn.Tanh(),
            ],
        ),
        "edge_selectors": torch_geometric.nn.Sequential(
            "nodes, edge_list, weights, B, T, t",
            [
                (
                    TemporalEdge(1),
                    "nodes, edge_list, weights, B, T, t -> edge_list, weights",
                )
            ],
        ),
    },
    "max_seq_len": seq_len,
}

CFG = base.CFG
CFG["ray"]["num_workers"] = 5
CFG["ray"]["num_gpus"] = 1.0
# For rollout workers
CFG["ray"]["num_gpus_per_worker"] = 0.2
CFG["ray"]["num_cpus_per_worker"] = 2
CFG["ray"]["train_batch_size"] = 1024
CFG["ray"]["rollout_fragment_length"] = seq_len

CFG["ray"]["model"] = base_graph
