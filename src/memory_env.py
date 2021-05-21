import gym
import numpy as np
from collections import OrderedDict

DEFAULT_CFG = {
    "dim": 1,
    "num_matches": 2,
    "num_cards": 8,
    "noise": 0,
    "mode": "view_flipped",  # Either view_all or view_current or view_flipped
    "discrete": True,
    "episode_length": 100,
    # Provide a small negative reward each timestep
    "negative_reward": False,
}


# IT DOESNT KNOW IF IT HAS A CARD FLIPPED!
# so make sure the obs space is num_matches large
# that shows all flipped cards
# OR connect edges to other flipped card?
class MemoryEnv(gym.Env):
    """Card memory environment.

    Obs space:
    All cards, with face down cards == 0

    Action space:
    Which card index to flip

    Reward:
    +1 for a match
    """

    def init_view_all(self):
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

    def init_view_current(self):
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
                        shape=(1,), high=self.cfg["num_cards"], low=0, dtype=np.float32
                    ),
                }
            )
        self.action_space = gym.spaces.Discrete(3)  # Left right flip

    def init_view_flipped(self):
        if self.cfg["discrete"]:
            assert self.cfg["dim"] == 1
            self.observation_space = gym.spaces.Dict(
                {
                    "card": gym.spaces.Discrete(
                        self.cfg["num_cards"] // self.cfg["num_matches"] + 1
                    ),
                    "flipped_cards": gym.spaces.MultiDiscrete(
                        [self.cfg["num_cards"] // self.cfg["num_matches"] + 1]
                        * (self.cfg["num_matches"])
                    ),
                    "flipped_pos": gym.spaces.MultiDiscrete(
                        [self.cfg["num_cards"]] * (self.cfg["num_matches"])
                    ),
                    "pointer_pos": gym.spaces.Discrete(self.cfg["num_cards"]),
                }
            )
        else:
            raise NotImplementedError()

        self.action_space = gym.spaces.Discrete(3)  # Left right flip

    def __init__(self, cfg={}):
        self.cfg = dict(DEFAULT_CFG, **cfg)
        assert self.cfg["num_cards"] % self.cfg["num_matches"] == 0
        assert self.cfg["mode"] in ["view_all", "view_current", "view_flipped"]

        if self.cfg["mode"] == "view_all":
            self.init_view_all()
        elif self.cfg["mode"] == "view_current":
            self.init_view_current()
        elif self.cfg["mode"] == "view_flipped":
            self.init_view_flipped()

    def get_obs(self, mode):
        visible_cards = np.ma.masked_array(self.cards)
        visible_cards[self.flipped == 0] = np.ma.masked
        visible_cards = visible_cards.filled(0)
        if mode == "view_all":
            return OrderedDict([("cards", visible_cards)])
        elif mode == "view_current":
            if self.cfg["discrete"]:
                pp = self.view_ptr
            else:
                # Box is expected to be an ndarray
                pp = np.array([self.view_ptr])
            return OrderedDict(
                [("card", visible_cards[self.view_ptr]), ("pointer_pos", pp)]
            )
        elif mode == "view_flipped":
            if self.cfg["discrete"]:
                # Only view flipped and unmatched cards
                flip_in_play_idx = self.flipped * ~self.matched
                flip_in_play = self.cards[flip_in_play_idx]
                # Zero pad so it's always the same size
                flip_in_play = np.pad(
                    flip_in_play,
                    (self.cfg["num_matches"] - len(flip_in_play), 0),
                    "constant",
                )
                [flip_in_play_pos] = flip_in_play_idx.nonzero()
                flip_in_play_pos = np.pad(
                    flip_in_play_pos,
                    # TODO should we use a nonzero value?
                    # network should be able to associate zero
                    # card value with zero pos value
                    (self.cfg["num_matches"] - len(flip_in_play_pos), 0),
                    "constant",
                )
                assert len(flip_in_play_pos) == len(flip_in_play)

                return OrderedDict(
                    [
                        ("card", visible_cards[self.view_ptr]),
                        ("flipped_cards", flip_in_play),
                        ("flipped_pos", flip_in_play_pos),
                        ("pointer_pos", self.view_ptr),
                    ]
                )
            else:
                raise NotImplementedError()

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
        if self.cfg["negative_reward"]:
            fail_reward = -1.0 / self.cfg["episode_length"]
        else:
            fail_reward = 0
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

        if self.cfg["mode"] in ["view_current", "view_flipped"]:
            self.view_ptr = 0

        self.tstep = 0

        return self.get_obs(self.cfg["mode"])

    def draw(self):
        if self.cfg["mode"] == "view_all":
            print(self.get_obs(self.cfg["mode"]))
        elif self.cfg["mode"] == "view_current":
            view = self.get_obs("view_all")["cards"]
            print(view)
            print(self.view_ptr)
        elif self.cfg["mode"] == "view_flipped":
            obs = self.get_obs(self.cfg["mode"])
            for k, v in obs.items():
                print(f"{k}: {v}")

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
