import gym
import random
import numpy as np
from collections import OrderedDict


DEFAULT_CFG = {
    "dim": 2,
    "max_items": 4,
    "max_queries": 2,
    # If we always want to recall the n'th elements
    "deterministic": False,
    # Use -1 to denote a false (random) element
    "determinstic_idxs": [],
}


class RecallEnv(gym.Env):
    """Recall memory environment.

    Obs -> [is_recall: [0,1] , item: [R^n]]
    Action -> [seen_previously: [0,1]]
    """

    def __init__(self, cfg={}):  # dim=8, max_items=32, max_queries=8):
        self.cfg = dict(DEFAULT_CFG, **cfg)
        self.max_items = self.cfg["max_items"]
        self.max_queries = self.cfg["max_queries"]
        self.curr_t = 0
        self.prev_queries = []
        self.prev_obs = []
        self.dim = self.cfg["dim"]
        """
        self.det = self.cfg['deterministic']
        self.det_idxs = self.cfg['deterministic_idxs']

        if self.det:
            assert len(self.det_idxs) == max_queries, "Max queries must equal len(deterministic_idxs)"
        """

        is_recall = gym.spaces.Discrete(2)
        item = gym.spaces.Box(
            shape=(self.dim,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )
        timestep = gym.spaces.Discrete(self.max_items + self.max_queries + 2)
        self.observation_space = gym.spaces.Dict(
            {"is_recall": is_recall, "item": item, "timestep": timestep}
        )

        # Seen previously, 1 is true and 0 is false
        self.action_space = gym.spaces.Discrete(2)

    def get_reward_range(self):
        return [0, 1]

    def get_done(self, obs):
        # +1 is for the weird mode between train/test
        if self.curr_t == self.max_items + self.max_queries + 1:
            return True

        return False

    def step(self, action):
        is_recall = -1
        item = None
        reward = 0

        # Train mode
        if self.curr_t < self.max_items - 1:
            is_recall = 0
            item = np.random.normal(size=self.dim)
            self.prev_queries.append(item)
        # First round of recall, we don't expect a valid action/reward here
        elif self.curr_t == self.max_items - 1:
            # Item is either in previous item or never seen
            is_recall = 1
            item = random.choice(
                [random.choice(self.prev_queries), np.random.normal(size=self.dim)]
            )
        # Recall/test mode
        else:
            # For next
            is_recall = 1
            last_obs = self.prev_obs[-1]
            is_prev_item = any(last_obs["item"] is q for q in self.prev_queries)
            if is_prev_item and action == 1:
                reward = 1
                self.correct += 1
                self.true_positives += 1
            elif not is_prev_item and action == 0:
                reward = 1
                self.correct += 1
                self.true_negatives += 1

            # Item is either in previous item or never seen
            item = random.choice(
                [random.choice(self.prev_queries), np.random.normal(size=self.dim)]
            )

        self.curr_t += 1
        obs = OrderedDict(
            [("is_recall", is_recall), ("item", item), ("timestep", self.curr_t)]
        )
        self.prev_obs.append(obs)

        done = self.get_done(obs)
        info = {
            "num_correct": self.correct,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
        }
        return obs, reward, done, info

    def reset(self):
        self.curr_t = 0
        self.correct = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.prev_queries.clear()
        self.prev_obs.clear()

        is_recall = 0
        item = np.random.normal(size=self.dim)
        self.prev_queries.append(item)
        obs = OrderedDict(
            [("is_recall", is_recall), ("item", item), ("timestep", self.curr_t)]
        )

        return obs

    def play(self):
        while True:
            action = 0
            done = False
            obs = self.reset()
            print(obs)
            while not done:
                obs, reward, done, info = self.step(action)
                print(obs)
                if obs["is_recall"]:
                    print("Reward is:", reward)
                    action = int(input("Enter action (0,1): "))
            print("Game over, restarting")


if __name__ == "__main__":
    e = RecallEnv()
    e.play()
