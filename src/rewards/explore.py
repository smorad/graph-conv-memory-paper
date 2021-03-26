import numpy as np


class ExplorationReward:
    """Provides a reward for aligning the agent motion vector with the
    goal vector. To inhibit moving back and forth to game the system,
    the reward is provided only if the agent is closer to the target
    than it has been before"""

    def __init__(self):
        self.rr = [0.0, 0.01]
        self.past_states = []

    def on_env_load(self, env):
        self.env = env

    def get_reward_range(self):
        return self.rr

    def get_reward(self, obs, grid_size=0.5):
        curr_state = self.env.habitat_env._sim.get_agent_state()

        for s in self.past_states:
            if np.linalg.norm(s.position - curr_state.position) < grid_size:
                return self.rr[0]

        self.past_states.append(curr_state)

        return self.rr[1]

    def reset(self):
        curr_state = self.env.habitat_env._sim.get_agent_state()
        self.past_states.clear()
        # Add first state so we dont get free reward
        self.past_states.append(curr_state)
