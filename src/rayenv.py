import os
import argparse
from collections import OrderedDict
import multiprocessing

import numpy as np
import torch
import cv2
import habitat
from habitat.utils.visualizations import maps
from habitat_baselines.common import obs_transformers
import gym
from gym.spaces import discrete
import ray

from server.render import RENDER_ROOT, CLIENT_LOCK
import util
from semantic_colors import COLORS64


class NavEnv(habitat.RLEnv):
    def __init__(self, cfg):
        self.visualize_lvl = cfg["visualize"]
        self.hab_cfg = habitat.get_config(config_paths=cfg["hab_cfg_path"])
        self.ray_cfg = cfg
        self.rewards = [reward_cls() for reward_cls in cfg["rewards"].values()]
        # Use only a subset of scenes to avoid memory pressure
        self.select_scene_subset(cfg, self.hab_cfg)
        # TODO: Set different random seeds for different workers (based on pid maybe)
        super().__init__(self.hab_cfg)

        # Each ray actor is a separate process
        # so we can use PIDs to determine which actor is running
        self.pid = os.getpid()
        # Setup debug rendering
        # worker_idx is populated by ray
        self.render_dir = f"{RENDER_ROOT}/{self.pid}"
        os.makedirs(self.render_dir, exist_ok=True)
        # Patch action space since habitat actions use custom spaces for some reason
        # TODO: these should translate for continuous/arbitrary action distribution
        self.action_space = discrete.Discrete(len(self.hab_cfg.TASK.POSSIBLE_ACTIONS))
        # Load reward functions
        [r.on_env_load(self) for r in self.rewards]
        # Observation transformers let us modify observations without
        # adding new sensors
        self.preprocessors = [
            preproc_cls(self) for preproc_cls in cfg["preprocessors"].values()
        ]
        self.observation_space = obs_transformers.apply_obs_transforms_obs_space(
            self.observation_space, self.preprocessors
        )

    def select_scene_subset(self, ray_cfg, hab_cfg):
        """Reduces the number of scenes for this environment. This helps alleviate memory
        pressure. Note that the left out scenes will be handled by other environments"""
        # Ray populates these
        # Worker starts at 1 not 0 (unless --local is passed)
        if not hasattr(ray_cfg, "worker_index"):
            return
        split = ray_cfg.worker_index - 1
        num_splits = ray_cfg.num_workers

        # Single process cases
        # 0 is --local
        # 1 is a single worker
        if ray_cfg.worker_index < 2:
            return
        # Worker includes the learner, so

        dataset = habitat.datasets.make_dataset(hab_cfg.DATASET.TYPE)
        scenes = hab_cfg.DATASET.CONTENT_SCENES

        # Not enough scenes, no need to split
        if len(scenes) < num_splits:
            return

        if "*" in hab_cfg.DATASET.CONTENT_SCENES:
            scenes = dataset.get_scenes_to_load(hab_cfg.DATASET)

        scene_splits = [[] for _ in range(num_splits)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

        hab_cfg.defrost()
        hab_cfg.DATASET.CONTENT_SCENES = scene_splits[split]
        hab_cfg.freeze()
        print(f"Env {split} loading scenes: {scene_splits[split]}")

    def emit_debug_imgs(self, obs, info, keys=[]):
        """Emit debug images to be served over the browser"""
        for key in keys:
            img = obs.get(key, None)
            if img is None:
                img = info.get(key, None)
            if img is None:
                continue

            if key == "depth":
                img = (img * 255).astype(np.uint8)
            elif key == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif key == "top_down_map":
                img = maps.colorize_draw_agent_and_fit_to_height(
                    img, self.hab_cfg.SIMULATOR.RGB_SENSOR.WIDTH
                )
            elif key == "semantic":
                sem = self.convert_sem_obs(obs)
                img = COLORS64[sem.flat].reshape(*sem.shape, 3)
            else:
                continue

            tmp_impath = f"{self.render_dir}/dbg_{key}.jpg.buf"
            impath = f"{self.render_dir}/dbg_{key}.jpg"
            _, buf = cv2.imencode(".jpg", img)
            buf.tofile(tmp_impath)
            # We do this so we don't accidentally load a half-written img
            os.replace(tmp_impath, impath)

    def emit_preproc_imgs(self, obs):
        for p in self.preprocessors:
            if hasattr(p, "visualize"):
                img = p.visualize()
                tmp_impath = f"{self.render_dir}/pp_{type(p).__name__}.jpg.buf"
                impath = f"{self.render_dir}/pp_{type(p).__name__}.jpg"
                _, buf = cv2.imencode(".jpg", img)
                buf.tofile(tmp_impath)
                # We do this so we don't accidentally load a half-written img
                os.replace(tmp_impath, impath)

    def convert_sem_obs(self, obs, in_place=False):
        """Convert the habitat semantic observation from
        instance IDs to category IDs"""
        # TODO: Paralellize this using worker pools
        if "semantic" not in obs:
            return
        sem = obs["semantic"]
        sem = self.instance_to_cat[sem.flat].reshape(sem.shape)
        return sem

    def obs_sanity_check(self, obs):
        for name, data in obs.items():
            assert (
                tuple(data.shape) == self.observation_space.spaces[name].shape
            ), f"Shape mismatch: declared: {tuple(data.shape)}, actual: {self.observation_space.spaces[name].shape}"
            sp = self.observation_space.spaces[name]
            assert (
                data.min() >= sp.low.flat[0]
            ), f"{name}: Min {data.min()} out of range {sp.low.flat[0]}"
            assert (
                data.max() <= sp.high.flat[0]
            ), f"{name}: Max {data.max()} out of range {sp.high.flat[0]}"

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Only visualize if someone is viewing via webbrowser
        self.maybe_viz(obs, info)
        obs = obs_transformers.apply_obs_transforms_batch(obs, self.preprocessors)
        # See https://discuss.ray.io/t/preprocessor-fails-on-observation-vector/614
        # order matters
        obs = OrderedDict((k, obs[k]) for k in self.observation_space.spaces)
        self.maybe_viz_pp(obs)
        self.obs_sanity_check(obs)
        return obs, reward, done, info

    # Habitat iface that we impl
    def get_reward_range(self):
        low, high = (0, 0)
        for reward_fn in self.rewards:
            l, h = reward_fn.get_reward_range()
            low += l
            high += h

        return [low, high]

    def get_reward(self, observations):
        rewards = [r.get_reward(observations) for r in self.rewards]
        return np.sum(rewards)

    def get_done(self, observations):
        return (
            self.habitat_env.episode_over or self.habitat_env.get_metrics()["success"]
        )

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def maybe_build_sem_lookup_table(self):
        # for some reason hasattr doesn't work here, try/except instead
        try:
            scene = self._env.sim.semantic_annotations()
            # See https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
            # for human-readable mapping
            self.instance_to_cat = np.zeros((len(scene.objects),), dtype=np.int)
            # 42 cats and 1 for error
            self.cat_to_str = np.zeros((43,), dtype=object)
            self.cat_to_str[-1] = "habitat_error"

            for obj in scene.objects:
                inst = int(obj.id.split("_")[-1])  # instance
                cat = obj.category.index()  # category
                cat_str = obj.category.name()  # str

                # Append -1 entries to end, space should exist for them
                if cat < 0:
                    cat = self.cat_to_str.size - 1

                self.instance_to_cat[inst] = cat
                self.cat_to_str[cat] = cat_str

        except NameError:
            pass

    def maybe_viz(self, obs, info={}):
        viz = []
        if CLIENT_LOCK.exists():
            if self.visualize_lvl >= 1:
                viz += ["rgb", "semantic", "depth"]
            if self.visualize_lvl >= 2:
                viz += ["top_down_map"]
        self.emit_debug_imgs(obs, info, viz)

    def maybe_viz_pp(self, obs):
        if CLIENT_LOCK.exists():
            if self.visualize_lvl >= 1:
                self.emit_preproc_imgs(obs)

    def reset(self):
        obs = super().reset()
        self.maybe_build_sem_lookup_table()
        # print('object', self.cat_to_str[obs['objectgoal']],obs['objectgoal'])
        # Reset reward functions
        [r.reset() for r in self.rewards]
        [p.reset() for p in self.preprocessors if hasattr(p, "reset")]
        self.maybe_viz(obs)
        obs = obs_transformers.apply_obs_transforms_batch(obs, self.preprocessors)
        # See https://discuss.ray.io/t/preprocessor-fails-on-observation-vector/614
        # order matters
        obs = OrderedDict((k, obs[k]) for k in self.observation_space.spaces)
        self.obs_sanity_check(obs)
        self.maybe_viz_pp(obs)
        # print('object', self.cat_to_str[obs['objectgoal']],obs['objectgoal'])
        return obs
