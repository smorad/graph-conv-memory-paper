import gym
import numpy as np

DEFAULT_CFG = {
    "dim": 1,
    "num_matches": 4,
    "num_cards": 52,
    "noise": 0,
    "mode": "view_current",  # Either view_all or view_current
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
            self.observation_space = gym.spaces.Dict(
                {
                    "card": gym.spaces.Box(
                        shape=(self.cfg["dim"],),
                        high=1,
                        low=0,
                        dtype=np.float32,
                    ),
                }
            )
            self.action_space = gym.spaces.Discrete(3)  # Left right flip

    def get_obs(self):
        visible_cards = np.ma.masked_array(self.cards)
        visible_cards[self.flipped == 0] = np.ma.masked
        visible_cards = visible_cards.filled(0)
        if self.cfg["mode"] == "view_all":
            return {"cards": visible_cards}
        else:
            return {"card": visible_cards[self.view_ptr]}

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
        reward = 0
        # If all three are flipped, mark as matched and give reward
        if self.flipped.sum() - self.matched.sum() == self.cfg["num_matches"]:
            self.matched[np.where(self.flipped)] = True
            reward += 1.0
        else:
            # Negative reward if we didnt match, to reward faster completion
            reward -= 0.01

        return reward

    def get_done(self):
        # If all are matched, we are done
        return np.all(self.matched)

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

        obs = self.get_obs()
        reward = self.get_reward()
        done = self.get_done()

        return obs, reward, done, {}

    def reset(self):
        num_unique_cards = self.cfg["num_cards"] // self.cfg["num_matches"]
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

        return self.get_obs()

    def play(self):
        while True:
            action = 0
            done = False
            obs = self.reset()
            print(obs)
            while not done:
                if self.cfg["mode"] == "view_all":
                    action = int(
                        input(f"Enter action (0...{self.cfg['num_cards'] - 1}): ")
                    )
                else:
                    action = int(input("Enter action (0,1,2): "))
                obs, reward, done, info = self.step(action)
                print(obs)
                print("Reward is:", reward)
            print("Game over, restarting")


if __name__ == "__main__":
    e = MemoryEnv()
    e.play()
