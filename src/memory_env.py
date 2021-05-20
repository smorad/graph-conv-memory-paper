import gym
import numpy as np
from collections import OrderedDict

DEFAULT_CFG = {
    "dim": 1,
    "num_matches": 2,
    "num_cards": 8,
    "noise": 0,
    "mode": "view_current",  # Either view_all or view_current
    "discrete": False,
    "episode_length": 100,
}


class MemoryEnv(gym.Env):
    """Card memory environment.

    Obs space:
    All cards, with face down cards == 0

    Action space:
    Which card index to flip

    Reward:
    +1 for a match
    """

    def __init__(self, cfg={}):
        self.cfg = dict(DEFAULT_CFG, **cfg)
        assert self.cfg["num_cards"] % self.cfg["num_matches"] == 0
        assert self.cfg["mode"] in ["view_all", "view_current"]

        if self.cfg["mode"] == "view_all":
            if self.cfg["discrete"]:
                self.observation_space = gym.spaces.Dict(
                    {
                        "cards": gym.spaces.MultiDiscrete(
                            [self.cfg["num_matches"] * self.cfg["dim"]]
                            * self.cfg["num_cards"]
                        ),
                    }
                )
            else:
                self.observation_space = gym.spaces.Dict(
                    {
                        "cards": gym.spaces.Box(
                            shape=(self.cfg["num_cards"], self.cfg["dim"]),
                            high=1,
                            low=0,
                            dtype=np.float32,
                        ),
                    }
                )
            self.action_space = gym.spaces.Discrete(self.cfg["num_cards"])
        else:
            if self.cfg["discrete"]:
                self.observation_space = gym.spaces.Dict(
                    {
                        "card": gym.spaces.Discrete(
                            self.cfg["num_cards"] // self.cfg["num_matches"] + 1
                        ),
                        "pointer_pos": gym.spaces.Discrete(self.cfg["num_cards"]),
                    }
                )
            else:
                self.observation_space = gym.spaces.Dict(
                    {
                        "card": gym.spaces.Box(
                            shape=(self.cfg["dim"],),
                            high=1,
                            low=0,
                            dtype=np.float32,
                        ),
                        "pointer_pos": gym.spaces.Box(
                            shape=(1,),
                            high=self.cfg["num_cards"],
                            low=0,
                            dtype=np.float32,
                        ),
                    }
                )
            self.action_space = gym.spaces.Discrete(3)  # Left right flip

    def get_obs(self, mode):
        visible_cards = np.ma.masked_array(self.cards)
        visible_cards[self.flipped == 0] = np.ma.masked
        visible_cards = visible_cards.filled(0)
        if mode == "view_all":
            return OrderedDict([("cards", visible_cards)])
        else:
            if self.cfg["discrete"]:
                pp = self.view_ptr
            else:
                # Box is expected to be an ndarray
                pp = np.array([self.view_ptr])
            return OrderedDict(
                [("card", visible_cards[self.view_ptr]), ("pointer_pos", pp)]
            )

    def flip_card(self, idx_to_flip):
        """Flips the card at the given index.
        Handles observation and state."""
        # If no other cards are flipped, flip the current card
        if not np.any(self.flipped):
            self.flipped[idx_to_flip] = True
        else:
            # Flipped idxs that are not yet matched
            flip_in_play = self.flipped * ~self.matched
            # If the selected card matches the prev flipped cards, leave them all flipped
            if np.all(self.match_ids[idx_to_flip] == self.match_ids[flip_in_play]):
                self.flipped[idx_to_flip] = True
            # if not, flip all cards except the current card
            # and solved cards back over
            else:
                self.flipped[flip_in_play] = False
                self.flipped[idx_to_flip] = True

    def get_reward(self):
        success_reward = 1.0 / (self.cfg["num_cards"] // self.cfg["num_matches"])
        fail_reward = -1.0 / self.cfg["episode_length"]
        # Let's bound reward to -1, 1
        reward = 0
        # If all three are flipped, mark as matched and give reward
        if self.flipped.sum() - self.matched.sum() == self.cfg["num_matches"]:
            self.matched[np.where(self.flipped)] = True
            reward += success_reward
        else:
            # Negative reward if we didnt match, to reward faster completion
            reward += fail_reward

        return reward

    def get_done(self):
        # If all are matched, we are done
        return np.all(self.matched) or self.tstep == self.cfg["episode_length"]

    def step(self, action):
        """Flip given card"""

        # Get which card to flip
        if self.cfg["mode"] == "view_all":
            self.flip_card(action)
        else:
            # Move left/right
            if action == 0:
                self.view_ptr = (self.view_ptr - 1) % self.cfg["num_cards"]
            elif action == 1:
                self.view_ptr = (self.view_ptr + 1) % self.cfg["num_cards"]
            # Flip
            if action == 2:
                self.flip_card(self.view_ptr)

        obs = self.get_obs(self.cfg["mode"])
        reward = self.get_reward()
        done = self.get_done()

        self.tstep += 1

        return obs, reward, done, {}

    def reset(self):
        num_unique_cards = self.cfg["num_cards"] // self.cfg["num_matches"]
        if self.cfg["discrete"]:
            assert self.cfg["dim"] == 1, "not implemented"
            # 0 is reserved for face-down card
            unique_cards = np.arange(1, num_unique_cards + 1)
        else:
            unique_cards = np.random.rand(num_unique_cards, self.cfg["dim"])
        self.cards = unique_cards.repeat(self.cfg["num_matches"], axis=0)
        # Each pair/tuple of matching cards has a unique id
        self.match_ids = np.arange(num_unique_cards).repeat(self.cfg["num_matches"])

        # Shuffle
        self.card_ids = np.random.permutation(len(self.cards))
        self.cards = self.cards[self.card_ids]
        self.match_ids = self.match_ids[self.card_ids]

        self.flipped = np.zeros(self.cfg["num_cards"], dtype=bool)
        self.matched = np.zeros(self.cfg["num_cards"], dtype=bool)

        if self.cfg["mode"] == "view_current":
            self.view_ptr = 0

        self.tstep = 0

        return self.get_obs(self.cfg["mode"])

    def draw(self):
        if self.cfg["mode"] == "view_all":
            print(self.get_obs(self.cfg["mode"]))
        else:
            view = self.get_obs("view_all")["cards"]
            print(view)
            print(self.view_ptr)

    def play(self):
        while True:
            action = 0
            done = False
            obs = self.reset()
            self.draw()
            running_reward = 0
            while not done:
                if self.cfg["mode"] == "view_all":
                    action = int(
                        input(f"Enter action (0...{self.cfg['num_cards'] - 1}): ")
                    )
                else:
                    action = int(input("Enter action (0,1,2): "))
                obs, reward, done, info = self.step(action)
                self.draw()
                print("Reward is:", reward)
                running_reward += reward
            print(f"Game over, total reward: {running_reward} restarting")

    def __repr__(self):
        return "MemoryEnv"


if __name__ == "__main__":
    e = MemoryEnv()
    e.play()
