import os
import argparse
import cv2
import multiprocessing
import habitat
from habitat.utils.visualizations import maps
import gym, ray
from ray.rllib.agents import ppo
from gym.spaces import discrete

from server.render import RENDER_ROOT


class NavEnv(habitat.RLEnv):

    # RLEnv exposes a gym iface with the following methods:
    '''
    def step(self, action):
        obs, reward, done, info = super().step(action) 
        return obs, reward, done, info

    def reset(self):
        return super().reset() 

    def render(self, mode='human'):
        return super().render() 

    def close(self):
        return super().close() 
    '''
    # Habitat override

    def __init__(self, cfg):
        hab_cfg = habitat.get_config(config_paths=cfg['hab_cfg_path'])
        # Ensure habitat only runs on the ray-allocated GPUs
        # to prevent GPU OOMs
        # This only works in non-local mode
        gpu_id = ray.get_gpu_ids()
        if len(gpu_id) == 1:
            print(f'Starting habitat instance on gpu {gpu_id[0]}')
            hab_cfg.defrost()
            hab_cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id[0]
            hab_cfg.freeze()
        elif len(gpu_id) == 0:
            print('No GPUs found but one is required.'
            ' We are likely running in local mode, using default gpu (likely gpu0).')
        else:
            raise NotImplementedError('Multiple GPUs allocated, we have only tested one per worker')
        super().__init__(hab_cfg)
        self.hab_cfg = hab_cfg

        # Patch action space since habitat actions use custom spaces for some reason
        # TODO: these should translate for continuous/arbitrary action distribution
        # Order: forward, stop, left, right
        self.action_space = discrete.Discrete(3)
        # Each ray actor is a separate process
        # so we can use PIDs to determine which actor is running
        self.pid = os.getpid()
        self.render_dir = f'{RENDER_ROOT}/{self.pid}'
        os.makedirs(self.render_dir, exist_ok=True)

    def emit_debug_img(self, obs):
        img = obs['rgb']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tmp_impath = f'{self.render_dir}/out.jpg.buf'
        impath = f'{self.render_dir}/out.jpg'
        _, buf = cv2.imencode(".jpg", img)
        buf.tofile(tmp_impath)
        # We do this so we don't accidentally load a half-written img
        os.replace(tmp_impath, impath)

    def emit_debug_map(self, info):
        w = self.hab_cfg.SIMULATOR.RGB_SENSOR.WIDTH
        img = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], w
        )
        tmp_impath = f'{self.render_dir}/map.jpg.buf'
        impath = f'{self.render_dir}/map.jpg'
        _, buf = cv2.imencode(".jpg", img)
        buf.tofile(tmp_impath)
        # We do this so we don't accidentally load a half-written img
        os.replace(tmp_impath, impath)

    def step(self, action):
        obs, reward, done, info = super().step(action) 
        self.emit_debug_img(obs)
        self.emit_debug_map(info)
        return obs, reward, done, info

    def action_space(self):
        # TODO: these should translate for continuous/arbitrary action distribution
        # Order: forward, stop, left, right
        return discrete.Discrete(3)

    # Habitat iface that we impl
    def get_reward_range(self):
        return [0, 1.0]

    def get_reward(self, observations):
        if self.habitat_env.get_metrics()['success']:
            return 1.0
        return 0.0

    def get_done(self, observations):
        # TODO get_metrics() success is 0.0 for some reason?
        return self.habitat_env.episode_over# or self.habitat_env.get_metrics()['success']

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
