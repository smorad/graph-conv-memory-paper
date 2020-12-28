import habitat
import gym, ray
from ray.rllib.agents import ppo


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

    def __init__(self, cfg):
        hab_cfg = habitat.get_config(config_paths=cfg['path'])
        return super().__init__(hab_cfg)

    # Habitat iface that we impl
    def get_reward_range(self):
        return [-1,1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


if __name__ == '__main__':
    ray.init()
    #hab_cfg = habitat.get_config(config_paths="/root/habitat-lab/configs/tasks/pointnav.yaml")
    hab_cfg_path = "/root/habitat-lab/configs/tasks/pointnav.yaml"
    ray_cfg = {'env_config': {'path': hab_cfg_path}, 
            'num_workers': 2,
            'num_gpus_per_worker': 2,
            'framework': 'torch'}
    trainer = ppo.PPOTrainer(env=NavEnv, config=ray_cfg)
    while True:
        print(trainer.train())
