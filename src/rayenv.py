import habitat
import gym, ray
from ray.rllib.agents import ppo
from gym.spaces import discrete


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
        hab_cfg = habitat.get_config(config_paths=cfg['path'])
        rv = super().__init__(hab_cfg)

        # Patch action space since habitat actions use custom spaces for some reason
        # TODO: these should translate for continuous/arbitrary action distribution
        # Order: forward, stop, left, right
        self.action_space = discrete.Discrete(4)

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
        return self.habitat_env.episode_over or self.habitat_env.get_metrics()['success']

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


if __name__ == '__main__':
    ray.init(dashboard_host='0.0.0.0')
    hab_cfg_path = "/root/habitat-lab/configs/tasks/pointnav.yaml"
    '''
    hab = NavEnv(cfg={'path': hab_cfg_path})
    import pdb; pdb.set_trace()
    '''
    ray_cfg = {'env_config': {'path': hab_cfg_path}, 
            'num_workers': 8,
            'num_gpus_per_worker': 0.5,
            'framework': 'torch'}
    trainer = ppo.PPOTrainer(env=NavEnv, config=ray_cfg)
    # Can access envs here: trainer.workers.local_worker().env
    #ray.util.pdb.set_trace()
    while True:
        print(trainer.train())
