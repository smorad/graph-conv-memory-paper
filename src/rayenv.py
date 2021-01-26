import os
import argparse
import numpy as np
import cv2
import multiprocessing
import habitat
from habitat.utils.visualizations import maps
import gym, ray
from ray.rllib.agents import ppo
from gym.spaces import discrete
from ray.rllib.env.external_env import ExternalEnv

from server.render import RENDER_ROOT, CLIENT_LOCK


class ExternalNavEnv(ExternalEnv):
    def __init__(self, hab_vec_env):
        self.env = hab_vec_env
        super().__init__(
                action_space=self.env.action_space,
                observation_space=self.env.observation_space
        )

    def run(self):
        eid = self.start_episode()
        obs = env.reset()
        while True:
            action = self.get_action(eid, obs)
            obs, reward, done, info = self.env.step(action)
            self.log_returns(eid, reward, info=info)
            if done:
                self.end_episode(eid, obs)
                obs = self.env.reset()
                eid = self.start_episode()



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

    def load_sem_sensor(self):
        import sensors.nn_semantic
        self.hab_cfg.defrost()
        self.hab_cfg.TASK.AGENT_POSITION_SENSOR = habitat.Config()
        self.hab_cfg.TASK.AGENT_POSITION_SENSOR.TYPE = "NNSemanticSensor"
        self.hab_cfg.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
        self.hab_cfg.freeze()

    def load_mesh_sem_sensor(self):
        import sensors.mesh_semantic
        self.hab_cfg.defrost()
        self.hab_cfg.TASK.SEMANTIC_MASK_SENSOR = habitat.Config()
        self.hab_cfg.TASK.SEMANTIC_MASK_SENSOR.TYPE = "SemanticMaskSensor"
        self.hab_cfg.TASK.SENSORS.append("SEMANTIC_MASK_SENSOR")
        self.hab_cfg.freeze()

    def set_gpu_id(self):
        # Ensure habitat only runs on the ray-allocated GPUs
        # to prevent GPU OOMs
        # This only works in non-local mode
        gpu_ids = ray.get_gpu_ids()
        if len(gpu_ids) == 1:
            print(f'Starting habitat instance on gpu {gpu_ids[0]}')
            self.hab_cfg.defrost()
            self.hab_cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_ids[0]
            self.hab_cfg.freeze()
        elif len(gpu_ids) == 0:
            print('No GPUs found but one is required.'
            ' We are likely running in local mode, using default gpu (likely gpu0).')
        else:
            raise NotImplementedError('Multiple GPUs allocated, we have only tested one per worker')

    def __init__(self, cfg):
        self.visualize = cfg['visualize']
        self.hab_cfg = habitat.get_config(config_paths=cfg['hab_cfg_path'])
        #self.load_sem_sensor()
        #self.load_mesh_sem_sensor()
        #self.set_gpu_id()
        super().__init__(self.hab_cfg)
        # Patch action space since habitat actions use custom spaces for some reason
        # TODO: these should translate for continuous/arbitrary action distribution
        self.action_space = discrete.Discrete(len(self.hab_cfg.TASK.POSSIBLE_ACTIONS))
        # Patch observation space because we modify semantic sensor
        # to produce object ids instead of instance ids

        # Each ray actor is a separate process
        # so we can use PIDs to determine which actor is running
        self.pid = os.getpid()
        self.render_dir = f'{RENDER_ROOT}/{self.pid}'
        os.makedirs(self.render_dir, exist_ok=True)

    def emit_debug_imgs(self, obs, info, keys=[]):
        '''Emit debug images to be served over the browser'''
        for key in keys:
            img = obs.get(key, None)
            if not img:
                img = info.get(key, None)
            if not img:
                break

            if key == 'depth':
                pass
            elif key == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif key == 'top_down_map':
                img = maps.colorize_draw_agent_and_fit_to_height(
                    img, self.hab_cfg.SIMULATOR.RGB_SENSOR.WIDTH                
                )
            else:
                break

            tmp_impath = f'{self.render_dir}/{key}.jpg.buf'
            impath = f'{self.render_dir}/{key}.jpg'
            _, buf = cv2.imencode(".jpg", img)
            buf.tofile(tmp_impath)
            # We do this so we don't accidentally load a half-written img
            os.replace(tmp_impath, impath)
            


    def emit_debug_depth(self, obs):
        img = obs['depth']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tmp_impath = f'{self.render_dir}/depth.jpg.buf'
        impath = f'{self.render_dir}/depth.jpg'
        _, buf = cv2.imencode(".jpg", img)
        buf.tofile(tmp_impath)
        # We do this so we don't accidentally load a half-written img
        os.replace(tmp_impath, impath)


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

    def convert_sem_obs(self, obs):
        '''Convert the habitat semantic observation from
        instance IDs to category IDs'''
        # TODO: Paralellize this using worker pools
        if not 'semantic' in obs:
            return
        for i in range(len(obs['semantic'].flat)):
            obs['semantic'].flat[i] = self.semantic_label_lookup(obs['semantic'].flat[i])

    def emit_semantic(self, obs, num_classes=40):
        # This needs to be really fast
        img = np.ones(*obs['semantic'].shape,3)
        norm = 255 // 40
        # for each channel
        img[:,:,0] = obs['semantic'] * norm % 255
        img[:,:,1] = obs['semantic'] * norm % 170
        img[:,:,2] = obs['semantic'] * norm % 85
        tmp_impath = f'{self.render_dir}/sem.jpg.buf'
        impath = f'{self.render_dir}/sem.jpg'
        _, buf = cv2.imencode(".jpg", img)
        buf.tofile(tmp_impath)
        # We do this so we don't accidentally load a half-written img
        os.replace(tmp_impath, impath)

    def step(self, action):
        obs, reward, done, info = super().step(action) 
        #import pdb; pdb.set_trace()
        #self.convert_sem_obs(obs)
        # Only visualize if someone is viewing via webbrowser
        viz = []
        if CLIENT_LOCK.exists():
            if self.visualize >= 1:
                viz += ['rgb', 'semantic']
            if self.visualize >= 2: 
                viz += ['top_down_map']
        self.emit_debug_imgs(obs, info, viz)
        return obs, reward, done, info

    # Habitat iface that we impl
    def get_reward_range(self):
        return [0, 1.0]

    def get_reward(self, observations):
        if self.habitat_env.get_metrics()['success']:
            return 1.0
        return 0.0

    def get_done(self, observations):
        return self.habitat_env.episode_over or self.habitat_env.get_metrics()['success']

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def reset(self):
        # for some reason hasattr doesn't work here, try/except instead
        try:
            scene = self._env.sim.semantic_annotations()
            # See https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
            # for human-readable mapping
            self._env.sim.semantic_label_lookup = {
                int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects
            }
            #TODO: We can't have negative numbers
            # find out what -1 actually means
            self._env.sim.semantic_label_lookup[-1] = 0
            #self.convert_sem_obs(obs)
        except NameError:
            pass
        obs = super().reset()
        #import pdb; pdb.set_trace()
        return obs


