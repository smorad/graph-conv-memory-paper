from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from cfg import train

CFG = train.CFG
CFG["ray"]["lr"] = tune.loguniform(1e-2, 1e-4)
CFG.update(
    {
        "tune": {
            "stop": {"training_iteration": 250},
            "num_samples": 40,
            "search_alg": HyperOptSearch(metric="episode_reward_mean", mode="max"),
        }
    }
)
