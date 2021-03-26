import numpy as np


class CollisionReward:
    """Provides a reward for aligning the agent motion vector with the
    goal vector. To inhibit moving back and forth to game the system,
    the reward is provided only if the agent is closer to the target
    than it has been before"""

    def __init__(self):
        self.rr = [-0.005, 0.0]
        self.past_states = []

    def on_env_load(self, env):
        self.env = env

    def get_reward_range(self):
        return self.rr

    def get_reward(self, obs, grid_size=0.5):
        if self.env.habitat_env._sim.previous_step_collided:
            return self.rr[0]
        return self.rr[1]

    def reset(self):
        pass
