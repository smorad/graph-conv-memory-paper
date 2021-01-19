import os
import argparse
import cv2
import multiprocessing
import habitat
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
        #print(f'Ray cfg: {cfg}')
        #print(f'Hab cfg: {hab_cfg}')
        rv = super().__init__(hab_cfg)

        # Patch action space since habitat actions use custom spaces for some reason
        # TODO: these should translate for continuous/arbitrary action distribution
        # Order: forward, stop, left, right
        self.action_space = discrete.Discrete(4)
        # Each ray actor is a separate process
        # so we can use PIDs to determine which actor is running
        self.pid = os.getpid()
        self.render_dir = f'{RENDER_ROOT}/{self.pid}'
        os.makedirs(self.render_dir, exist_ok=True)

    def emit_debug_img(self, obs):
        img = obs['rgb']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tmp_impath = f'{self.render_dir}/_out.jpg'
        impath = f'{self.render_dir}/out.jpg'
        cv2.imwrite(tmp_impath, img)
        # We do this so we don't accidentally load a half-written img
        os.replace(tmp_impath, impath)

    def step(self, action):
        obs, reward, done, info = super().step(action) 
        self.emit_debug_img(obs)
        return obs, reward, done, info

    def action_space(self):
        # TODO: these should translate for continuous/arbitrary action distribution
        # Order: forward, stop, left, right
        return discrete.Discrete(4)

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
