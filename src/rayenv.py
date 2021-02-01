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


class NavEnv(habitat.RLEnv):
    def __init__(self, cfg):
        self.visualize = cfg["visualize"]
        self.hab_cfg = habitat.get_config(config_paths=cfg["hab_cfg_path"])
        # TODO: Set different random seeds for different workers (based on pid maybe)
        super().__init__(self.hab_cfg)

        # Each ray actor is a separate process
        # so we can use PIDs to determine which actor is running
        self.pid = os.getpid()
        # Setup debug rendering
        self.render_dir = f"{RENDER_ROOT}/{self.pid}"
        os.makedirs(self.render_dir, exist_ok=True)
        # Patch action space since habitat actions use custom spaces for some reason
        # TODO: these should translate for continuous/arbitrary action distribution
        self.action_space = discrete.Discrete(len(self.hab_cfg.TASK.POSSIBLE_ACTIONS))
        # Observation transformers let us modify observations without
        # adding new sensors
        # TODO: SemanticMask adds takes startup time from 20s to 160s
        # and OOMs gpus. Likely due to atari preprocessor
        self.preprocessors = [
            util.load_class(cfg["preprocessors"], k)(self) for k in cfg["preprocessors"]
        ]
        self.observation_space = obs_transformers.apply_obs_transforms_obs_space(
            self.observation_space, self.preprocessors
        )

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
                # This needs to be really fast
                img = np.ones((*sem.shape, 3), dtype=np.uint8)
                norm = 255 // 40
                min_px = 20  # so things aren't too dark
                # for each channel
                img[:, :, 0] = min_px + sem * norm % 255
                img[:, :, 1] = min_px + sem * norm % 170
                img[:, :, 2] = min_px + sem * norm % 85
            else:
                continue

            tmp_impath = f"{self.render_dir}/{key}.jpg.buf"
            impath = f"{self.render_dir}/{key}.jpg"
            _, buf = cv2.imencode(".jpg", img)
            buf.tofile(tmp_impath)
            # We do this so we don't accidentally load a half-written img
            os.replace(tmp_impath, impath)

    def convert_sem_obs(self, obs):
        """Convert the habitat semantic observation from
        instance IDs to category IDs"""
        # TODO: Paralellize this using worker pools
        if "semantic" not in obs:
            return
        sem = obs["semantic"].copy()
        for i in range(len(sem.flat)):
            sem.flat[i] = self.semantic_label_lookup[sem.flat[i]]
        return sem

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Only visualize if someone is viewing via webbrowser
        viz = []
        if CLIENT_LOCK.exists():
            if self.visualize >= 1:
                viz += ["rgb", "semantic", "depth"]
            if self.visualize >= 2:
                viz += ["top_down_map"]
        self.emit_debug_imgs(obs, info, viz)

        obs = obs_transformers.apply_obs_transforms_batch(obs, self.preprocessors)
        # See https://discuss.ray.io/t/preprocessor-fails-on-observation-vector/614
        # order matters
        obs = OrderedDict((k, obs[k]) for k in self.observation_space.spaces)
        return obs, reward, done, info

    # Habitat iface that we impl
    def get_reward_range(self):
        return [0, 1.0]

    def get_reward(self, observations):
        if self.habitat_env.get_metrics()["success"]:
            return 1.0
        return 0.0

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
            semantic_label_dict = {
                int(obj.id.split("_")[-1]): obj.category.index()
                for obj in scene.objects
            }
            # TODO: We can't have negative numbers
            # find out what -1 actually means
            semantic_label_dict[-1] = 0
            self.semantic_label_lookup = np.zeros(
                (len(semantic_label_dict),), dtype=np.int32
            )
            for inst, cat in semantic_label_dict.items():
                self.semantic_label_lookup[inst] = cat
        except NameError:
            pass

    def reset(self):
        self.maybe_build_sem_lookup_table()
        obs = super().reset()
        obs = obs_transformers.apply_obs_transforms_batch(obs, self.preprocessors)
        # See https://discuss.ray.io/t/preprocessor-fails-on-observation-vector/614
        # order matters
        obs = OrderedDict((k, obs[k]) for k in self.observation_space.spaces)
        return obs
