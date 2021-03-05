from cfg import graph
from torch_geometric.nn import GATConv

# Run with:
# RAY_PDB=1 python3 start.py cfg/graph.debug.py --local
# see below why

# Likely you want to run with RAY_PDB=1 start cfg/graph_debug.py
# for postmortem debugging

CFG = graph.CFG
# workers == 0 only works with --local
# use workers == 1 if you want to avoid local mode
# this is due to opengl/cuda contexts used by env
CFG["ray"]["num_workers"] = 0
CFG["tune"]["stop"] = {"training_iteration": 1}
CFG["ray"]["train_batch_size"] = 8
CFG["ray"]["rollout_fragment_length"] = 8

CFG["ray"]["model"]["custom_model_config"]["gcn_conv_type"] = GATConv
CFG["ray"]["model"]["custom_model_config"]["gcn_num_attn_heads"] = 8
