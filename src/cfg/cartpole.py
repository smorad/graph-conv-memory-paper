import os

from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env, grid_search

from rewards.basic import BasicReward
from rewards.path import PathReward

from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole

from models.ray_graph import RayObsGraph


register_env(StatelessCartPole.__name__, StatelessCartPole)
cfg_dir = os.path.abspath(os.path.dirname(__file__))

models = [
    # Base model
    {
        "framestack": False,
    },
    # Framestack
    {"num_framestacks": 4},
    # LSTM
    {"use_lstm": True, "max_seq_len": 32, "lstm_cell_size": 32},
    # Ours
    {
        "max_seq_len": 32,
        "custom_model": RayObsGraph,
        "custom_model_config": {
            "graph_size": 32,
            "gcn_output_size": 8,
            "gcn_hidden_size": 32,
        },
    },
]


CFG = {
    # Our specific trainer type
    "ray_trainer": ImpalaTrainer,
    # Ray specific config sent to ray.tune or ray.rllib trainer
    "ray": {
        # These are rllib/ray specific
        "env_config": {},
        "framework": "torch",
        "model": grid_search(models[-1:]),
        "num_cpus_per_worker": 16,
        "num_envs_per_worker": 8,
        "num_gpus_per_worker": 0.1,
        "num_gpus": 1,
        "env": StatelessCartPole,
        "entropy_coeff": 0.001,
        "vf_loss_coeff": 1e-5,
        "gamma": 0.99,
    },
    "tune": {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 5e6},
    },
}
