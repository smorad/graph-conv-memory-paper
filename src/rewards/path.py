import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


class PathReward:
    """Provides a reward for aligning the agent motion vector with the
    goal vector. To inhibit moving back and forth to game the system,
    the reward is provided only if the agent is closer to the target
    than it has been before"""

    def __init__(self):
        self.rr = [0.0, 0.01]

    def on_env_load(self, env):
        self.env = env
        self.follower = ShortestPathFollower(
            self.env.habitat_env.sim,
            self.env.episodes[0].goals[0].radius,
            return_one_hot=False,
        )

    def get_reward_range(self):
        return self.rr

    def get_reward(self, obs):
        curr_state = self.env.habitat_env._sim.get_agent_state()
        goal_state = self.env.habitat_env.current_episode.goals[0]
        dist2goal = np.linalg.norm(goal_state.position - curr_state.position)
        # Prevent jittering, only reward if making progress
        if dist2goal >= self.closest:
            return 0.0

        self.closest = dist2goal

        # Cosine distance between current move and goal vector
        move_vec = curr_state.position - self.last_state.position
        move_vec /= np.linalg.norm(move_vec)
        goal_vec = goal_state.position - self.last_state.position
        goal_vec /= np.linalg.norm(goal_vec)
        reward_scale = max(0.0, np.dot(move_vec, goal_vec))
        reward = reward_scale * self.rr[1]

        self.last_state = curr_state
        return reward

    def reset(self):
        self.last_state = self.env.habitat_env._sim.get_agent_state()
        self.closest = np.linalg.norm(
            self.env.habitat_env.current_episode.goals[0].position
            - self.last_state.position
        )
