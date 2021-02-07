class BasicReward:
    def __init__(self):
        pass

    def on_env_load(self, env):
        self.env = env

    def get_reward_range(self):
        return [0.0, 1.0]

    def get_reward(self, obs):
        r = 0.0
        if self.env.habitat_env.get_metrics()["success"]:
            r += 1.0
        return r

    def reset(self):
        pass
