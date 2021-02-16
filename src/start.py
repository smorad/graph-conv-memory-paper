import argparse
import atexit
import importlib
import subprocess
import multiprocessing
import json
import time
import shutil
import ray
import server.render
import os
import inspect
import torch
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.rollout import rollout, RolloutSaver

import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("master_cfg", help="Path to the master .json cfg")
    parser.add_argument("--mode", help="Train, eval, or human", default="train")
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
    parser.add_argument(
        "--export-torch",
        "-e",
        default=None,
        type=str,
        help="Export saved rllib model as torch to this path",
    )
    args = parser.parse_args()
    return args


def start_tb():
    tb_proc = subprocess.Popen(
        ["tensorboard", "--bind_all", "--logdir", "/root/ray_results"]
    )
    atexit.register(tb_proc.terminate)
    return tb_proc


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
    cfg["ray"]["callbacks"] = util.load_class(cfg, "callback")
    if "model" in cfg:
        model_class = util.load_class(cfg, "model")
        ModelCatalog.register_custom_model(model_class.__name__, model_class)
        print(
            f"Starting: trainer: {trainer_class.__name__}: "
            f"env: {env_class.__name__} model: {model_class.__name__}"
        )
        cfg["ray"]["model"]["custom_model"] = model_class
    else:
        print(
            f"Starting: trainer: {trainer_class.__name__}: "
            f"env: {env_class.__name__} model: RAY DEFAULT"
        )
    start_tb()
    trainer = trainer_class(env=env_class, config=cfg["ray"])
    epoch = 0
    while True:
        print(f"Epoch: {epoch}")
        start_t = time.time()
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


def evaluate(args, cfg):
    ray.init(
        dashboard_host="0.0.0.0",
        local_mode=args.local,
        object_store_memory=args.object_store_mem,
    )

    env_class = util.load_class(cfg, "env_wrapper")
    trainer_class = util.load_class(cfg, "trainer")
    cfg["ray"]["callbacks"] = util.load_class(cfg, "callback")
    if "model" in cfg:
        model_class = util.load_class(cfg, "model")
        ModelCatalog.register_custom_model(model_class.__name__, model_class)
        print(
            f"Starting: trainer: {trainer_class.__name__}: "
            f"env: {env_class.__name__} model: {model_class.__name__}"
        )
        cfg["ray"]["model"]["custom_model"] = model_class
    else:
        print(
            f"Starting: trainer: {trainer_class.__name__}: "
            f"env: {env_class.__name__} model: RAY DEFAULT"
        )
    start_tb()
    trainer = trainer_class(env=env_class, config=cfg["ray"])

    if "checkpoint" not in cfg:
        raise Exception("Checkpoint required for evaluation")
    if "num_episodes" not in cfg:
        raise Exception("Did you forget to set `num_episodes`?")

    trainer.restore(cfg["checkpoint"])
    if args.export_torch:
        print(f"Exporting torch checkpoint to {args.export_torch}")
        torch.save(trainer.get_policy().model, args.export_torch)
    with RolloutSaver(
        outfile="/dev/shm/eval_output.pkl",
        use_shelve=False,
        target_episodes=cfg["num_episodes"],
        save_info=False,
    ) as rs:
        rollout(
            agent=trainer,
            env_name=None,
            num_steps=None,
            num_episodes=cfg["num_episodes"],
            saver=rs,
        )

    trainer.stop()


def human(args, cfg, act_q, resp_q):
    env_class = util.load_class(cfg, "env_wrapper")
    env = env_class(cfg=cfg["ray"]["env_config"])
    ep = 0
    done = False
    action_map = {"w": 0, "a": 1, "d": 2, "q": 3, "e": 4, " ": 5}
    while True:
        print(f"Episode: {ep}")
        env.reset()
        cum_reward = 0
        while not done:
            # Episode
            user_action = act_q.get()
            if user_action not in action_map:
                print(f"Invalid action {user_action}")
                continue

            env_action = action_map[user_action]
            obs, reward, done, info = env.step(env_action)
            cum_reward += reward
            info_wo_map = info.copy()
            # Not JSON serializable
            info_wo_map.pop("top_down_map", None)
            resp_q.put(
                {
                    "reward": reward,
                    "success": info["success"],
                    "target": env.cat_to_str[obs["objectgoal"][0]],
                    "target_in_view": bool(obs.get("target_in_view", [-1])[0]),
                    # **info_wo_map,
                }
            )
            # print(reward, done, info)

        print(f"Episode {ep} done, cum. reward {cum_reward}")
        ep += 1
        done = False


def main():
    args = get_args()
    cfg = load_master_cfg(args.master_cfg)

    cfg["ray"]["env_config"]["visualize"] = args.visualize
    # Rendering obs to website for remote debugging
    shutil.rmtree(server.render.RENDER_ROOT, ignore_errors=True)
    os.makedirs(server.render.RENDER_ROOT, exist_ok=True)
    action_q = multiprocessing.Queue()
    resp_q = multiprocessing.Queue()
    render_server = multiprocessing.Process(
        target=server.render.main, args=(action_q, resp_q)
    )
    render_server.start()

    if args.mode == "train":
        train(args, cfg)
    elif args.mode == "eval":
        evaluate(args, cfg)
    elif args.mode == "human":
        human(args, cfg, action_q, resp_q)
    else:
        raise NotImplementedError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
