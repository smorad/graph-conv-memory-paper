from cfg import base
from models.ray_graph import RayObsGraph

from ray.rllib.agents.ppo.ddppo import DDPPOTrainer

CFG = base.CFG
CFG["ray_trainer"] = DDPPOTrainer
CFG["ray"]["model"]["custom_model"] = RayObsGraph
CFG["ray"]["model"]["custom_model_config"] = {
    "graph_size": 32,
    "gcn_output_size": 128,
    "gcn_hidden_size": 256,
}

# This must be zero for ddppo
del CFG["ray"]["num_gpus"]
del CFG["ray"]["train_batch_size"]

# For rollout workers
# Each env gets its own learner
# but due to opengl/cuda/forking we can only
# have one env per worker
CFG["ray"]["num_workers"] = 24
CFG["ray"]["num_envs_per_worker"] = 1
CFG["ray"]["num_gpus_per_worker"] = 0.16
CFG["ray"]["num_cpus_per_worker"] = 2

CFG["ray"]["rollout_fragment_length"] = 100
