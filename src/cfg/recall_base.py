import os
from ray.tune import register_env
from recall_env import RecallEnv
from ray.rllib.agents.impala import ImpalaTrainer
from models.ray_graph import RayObsGraph
from ray.tune import grid_search

from models.edge_selectors.bernoulli import BernoulliEdge
from models.edge_selectors.temporal import TemporalBackedge


register_env(RecallEnv.__name__, RecallEnv)

base_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": 17,
        "gcn_output_size": 16,
        "gcn_hidden_size": 16,
        "gcn_num_layers": 8,
        "edge_selectors": [],
    },
    "max_seq_len": 8,
}

temporal_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": 17,
        "gcn_output_size": 16,
        "gcn_hidden_size": 16,
        "gcn_num_layers": 8,
        "edge_selectors": [TemporalBackedge],
    },
    "max_seq_len": 8,
}

bernoulli_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": 17,
        "gcn_output_size": 16,
        "gcn_hidden_size": 16,
        "gcn_num_layers": 8,
        "edge_selectors": [BernoulliEdge],
    },
    "max_seq_len": 8,
}

rnn_model = {"use_lstm": True, "max_seq_len": 8, "lstm_cell_size": 16}

models = [rnn_model, base_model, temporal_model, bernoulli_model]

CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        "env_config": {"dim": 4, "max_items": 4, "max_queries": 4},
        # These are rllib/ray specific
        "framework": "torch",
        "model": grid_search(models),
        "num_workers": 2,
        # Total GPU usage: num_gpus (trainer proc) + num_gpus_per_worker (workers)
        "num_cpus_per_worker": 4,
        "num_envs_per_worker": 1,
        # this corresponds to the number of learner GPUs used,
        # not the total used for the environments/rollouts
        "num_gpus": 1,
        # Size of batches (in timesteps) placed in the learner queue
        "rollout_fragment_length": 16,
        # Total number of timesteps to train per batch
        "train_batch_size": 512,
        "lr": 0.0001,
        "env": RecallEnv.__name__,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"training_iteration": 250},
    },
    # Env to be loaded when mode == human
    "human_env": RecallEnv,
}

if os.environ.get("DEBUG", False):
    print("-------DEBUG MODE---------")
    CFG["ray"]["model"]["custom_model_config"]["edge_selectors"] = [TemporalBackedge]
    CFG["ray"]["model"]["custom_model_config"]["export_gradients"] = True
    # CFG['ray']['model']['custom_model_config']
    # CFG['ray']['num_envs_per_worker'] = 1
    # CFG["ray"]["num_workers"] = 0
    # CFG['ray']['train_batch_size'] = 64
    # CFG['ray']['rollout_fragment_length'] = 32
