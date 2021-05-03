import os
from typing import Dict, Any

from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env, grid_search

from rewards.basic import BasicReward
from rewards.path import PathReward

from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole

from models.ray_graph import RayObsGraph
from models.edge_selectors.temporal import TemporalBackedge

from copy import deepcopy
import torch_geometric
import torch


register_env(StatelessCartPole.__name__, StatelessCartPole)
cfg_dir = os.path.abspath(os.path.dirname(__file__))

hidden = 8
seq_len = 200
gsize = 6  # seq_len + 1


no_mem: Dict[str, Any] = {
    "fcnet_hiddens": [hidden],
    "fcnet_activation": "tanh",
}
rnn_model = {
    **no_mem,
    "use_lstm": True,
    "max_seq_len": seq_len,
    "lstm_cell_size": hidden,
    # "lstm_use_prev_action": True,
}  # type: ignore
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
base_model = {
    "custom_model": RayObsGraph,
    "custom_model_config": {
        "graph_size": gsize,
        "gnn_input_size": hidden,
        "gnn_output_size": hidden,
        "gnn": dgc,
        # "use_prev_action": True,
    },
    "max_seq_len": seq_len,
}
temporal_model = deepcopy(base_model)
temporal_model["custom_model_config"]["edge_selectors"] = TemporalBackedge()

models = [
    # Ours
    temporal_model,
    # LSTM
    rnn_model,
    no_mem,
]


CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        # These are rllib/ray specific
        "env_config": {},
        "framework": "torch",
        "model": grid_search(models),
        "num_workers": 2,
        "num_cpus_per_worker": 2,
        # "num_envs_per_worker": 4,
        # "num_gpus_per_worker": 0.1,
        "num_gpus": 1,
        "env": StatelessCartPole,
        # "entropy_coeff": 0.001,
        "vf_loss_coeff": 0.01,
        "horizon": seq_len,
        # "batch_mode": "complete_episodes",
        # "vtrace": False,
        # "lr": 0.0005,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 5e6},
    },
}
