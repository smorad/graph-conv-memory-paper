import argparse
import importlib
import multiprocessing
import json
import time
import shutil
import ray
import server.render
import os
import inspect
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("master_cfg", help="Path to the master .json cfg")
    parser.add_argument("--mode", help="Train or eval", default="train")
    parser.add_argument(
        "--object-store-mem", help="Size of object store in bytes", default=3e10
    )
    parser.add_argument(
        "--local", action="store_true", default=False, help="Run ray in local mode"
    )
    parser.add_argument(
        "--visualize",
        "-v",
        default=1,
        type=int,
        help="Visualization level, higher == more visualization == slower",
    )
    args = parser.parse_args()
    return args


def load_master_cfg(path):
    """Master cfg should be json. It should be of format
    {
        ray_cfg: {
            env_config: {
                '...', # path to habitat cfg
            }
            ...
        },
        env_wrapper: "module_here.ClassGoesHere"
        ...
    }
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def train(args, cfg):
    ray.init(
        dashboard_host="0.0.0.0",
        local_mode=args.local,
        object_store_memory=args.object_store_mem,
    )

    env_class = util.load_class(cfg, "env_wrapper")
    trainer_class = util.load_class(cfg, "trainer")
    if "model" in cfg:
        model_class = util.load_class(cfg, "model")
        ModelCatalog.register_custom_model(model_class.__name__, model_class)
        print(
            f"Starting: trainer: {trainer_class.__name__}: "
            f"env: {env_class.__name__} model: {model_class.__name__}"
        )
    else:
        print(
            f"Starting: trainer: {trainer_class.__name__}: "
            f"env: {env_class.__name__} model: RAY DEFAULT"
        )
    trainer = trainer_class(env=env_class, config=cfg["ray"])
    epoch = 0
    start_t = time.time()
    while True:
        print(f"Epoch: {epoch}")
        epoch_results = trainer.train()
        num_steps = sum(epoch_results["hist_stats"]["episode_lengths"])
        print(pretty_print(epoch_results))
        epoch_t = time.time() - start_t
        print(f"Epoch {epoch} done in {epoch_t:.2f}s, {num_steps / epoch_t:.2f} fps")
        if epoch % 50 == 0:
            cpt = trainer.save()
            print(f"Saved to {cpt}")

        epoch += 1
        if epoch >= cfg.get("max_epochs", float("inf")):
            print(f"Trained for {epoch} epochs, terminating")
            break


def eval(args, cfg):
    pass


def main():
    args = get_args()
    cfg = load_master_cfg(args.master_cfg)

    cfg["ray"]["env_config"]["visualize"] = args.visualize
    # Rendering obs to website for remote debugging
    shutil.rmtree(server.render.RENDER_ROOT, ignore_errors=True)
    os.makedirs(server.render.RENDER_ROOT, exist_ok=True)
    render_server = multiprocessing.Process(
        target=server.render.main,
    )
    render_server.start()

    if args.mode == "train":
        train(args, cfg)
    elif args.mode == "eval":
        eval(args, cfg)
    else:
        raise NotImplementedError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
