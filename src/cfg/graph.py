from cfg import base
from models.ray_graph import RayObsGraph


CFG = base.CFG
CFG["ray"]["model"]["custom_model"] = RayObsGraph
CFG["ray"]["num_workers"] = 15
CFG["ray"]["model"]["custom_model_config"] = {
    "graph_size": 32,
    "gcn_output_size": 256,
    "gcn_hidden_size": 256,
    "gcn_num_layers": 3,
}
# How many past states are used for training
# this should likely be at least `graph_size`
CFG["ray"]["model"]["max_seq_len"] = 32
# this corresponds to the number of learner GPUs used,
# not the total used for the environments/rollouts
# Since this is the bottleneck, we let it use an entire GPU
CFG["ray"]["num_gpus"] = 1

# For rollout workers
CFG["ray"]["num_gpus_per_worker"] = 0.2
CFG["ray"]["num_cpus_per_worker"] = 4

# At batch sizes of 1024 and 2048, GPU learn time is roughly the same per sample
CFG["ray"]["train_batch_size"] = 2048
CFG["ray"]["rollout_fragment_length"] = 128
