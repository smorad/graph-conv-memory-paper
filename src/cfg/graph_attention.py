from cfg import graph
from torch_geometric.nn import GATConv


CFG = graph.CFG
CFG["ray"]["model"]["custom_model_config"]["gcn_conv_type"] = GATConv
CFG["ray"]["model"]["custom_model_config"]["gcn_num_attn_heads"] = 8
CFG["ray"]["num_workers"] = 6
CFG["ray"]["num_gpus_per_worker"] = 0.3
