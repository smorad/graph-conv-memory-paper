import argparse
import importlib.util
import atexit
import importlib
import subprocess
import multiprocessing
import time
import shutil
import ray
import server.render
import os
import inspect
import torch
import torch.autograd.profiler as profiler
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.rollout import rollout, RolloutSaver
from ray import tune
import hyperopt
from ray.tune.suggest.hyperopt import HyperOptSearch

import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("master_cfg", help="Path to the master .py config file")
    parser.add_argument("--mode", help="Train, eval, or human", default="train")
    parser.add_argument(
        "--object-store-mem",
        help="Size of object store in bytes. If this is too small ray will complain",
        default=3e10,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run ray in local mode. Useful for debugging, allows the use of local pdb",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        default=1,
        type=int,
        help="Visualization level, higher == more visualization == slower. Set to 0 if not using NavEnv or you will get an error.",
    )
    parser.add_argument(
        "--export-torch",
        "-e",
        default=None,
        nargs=2,
        help="Export saved rllib model from {ray_checkpoint} to {path}. Note {ray_checkpoint} is the path to the file, e.g. /root/vnav/data/IMPALA_2021-03-06_11-59-34/IMPALA_NavEnv_6aebc_00000_0_z_dim=256_2021-03-06_11-59-34/checkpoint_1999/checkpoint-1999",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Resume training after interruption, using the top-level dir in ~/ray_results (e.g. IMPALA_2021-03-06_11-59-34",
    )
    parser.add_argument(
        "--resume-error",
        default=None,
        type=str,
        help="Resume training after interruption, using the top-level dir in ~/ray_results (e.g. IMPALA_2021-03-06_11-59-34. This mode will retry failed trials only.",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use torch profiler to measure gpu usage over a trial",
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
    """Given a path to a .py config, load it and return the dict <path>.CFG"""
    spec = importlib.util.spec_from_file_location("dynamic_config", path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    return cfg_mod.CFG


def get_trial_str(trial):
    env = trial.config["env"]
    edge_selectors = (
        trial.config["model"].get("custom_model_config", {}).get("edge_selectors", None)
    )
    if edge_selectors:
        edge_selectors = [e.__class__.__name__ for e in edge_selectors]

    custom_model = trial.config["model"].get("custom_model", None)
    if custom_model:
        custom_model = custom_model.__name__

    if custom_model and edge_selectors:
        model = "_".join([custom_model, *edge_selectors])
    else:
        model = trial.config["model"]

    return f"{env}-{trial.trial_id}-{model}"


def train(args, cfg):
    ray.init(
        dashboard_host="0.0.0.0",
        local_mode=args.local,
        object_store_memory=args.object_store_mem,
    )

    start_tb()
    reporter = tune.CLIReporter(
        metric_columns=["episode_reward_mean", "training_iteration", "timers"],
        parameter_columns=[],
    )

    tune_cfg = cfg.get("tune", {})
    stop_cond = tune_cfg.get("stop", {})
    goal_metric = tune_cfg.get(
        "goal_metric", {"metric": "episode_reward_mean", "mode": "max"}
    )
    num_samples = tune_cfg.get("num_samples", 1)
    search_alg = tune_cfg.get("search_alg", None)
    # Required for checkpoint logic
    cpt_metric = goal_metric["metric"]
    if goal_metric["mode"] == "min":
        cpt_metric = "min-" + cpt_metric

    resume = args.resume_error or args.resume

    if args.profile:
        prof = profiler.profile(record_shapes=True, profile_memory=True)
        prof.__enter__()

    analysis = tune.run(
        cfg["ray_trainer"],
        config=cfg["ray"],
        trial_name_creator=get_trial_str,
        search_alg=search_alg,
        num_samples=num_samples,
        progress_reporter=reporter,
        stop=stop_cond,
        keep_checkpoints_num=5,
        checkpoint_score_attr=cpt_metric,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        metric=goal_metric["metric"],
        mode=goal_metric["mode"],
        resume=bool(resume),
        run_errored_only=bool(args.resume_error),
        name=resume,
        log_to_file=True,
    )

    if args.profile:
        prof.__exit__(None, None, None)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    print(f"Best trial: {analysis.best_trial}")


def export_torch(args, cfg):
    """Convert a ray checkpoint to a torch checkpoint"""
    ray.init(
        dashboard_host="0.0.0.0",
        local_mode=args.local,
        object_store_memory=args.object_store_mem,
    )
    # Don't load multiple envs for the sake of time
    cfg["ray"]["num_workers"] = 0
    train = cfg["ray_trainer"](env=cfg["ray"]["env"], config=cfg["ray"])
    train.restore(args.export_torch[0])
    torch.save(train.get_policy().model, args.export_torch[1])


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
    env = cfg["human_env"](cfg=cfg["ray"]["env_config"])
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
    if args.visualize > 0:
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
    elif args.mode == "tune":
        tune(args, cfg)
    elif args.mode == "eval":
        evaluate(args, cfg)
    elif args.mode == "export":
        export_torch(args, cfg)
    elif args.mode == "human":
        human(args, cfg, action_q, resp_q)
    else:
        raise NotImplementedError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
