import itertools
import random

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

from ..utils import HistoryTracker, describe_act

# Each integer action (0..5) corresponds to a move (src, dst) meaning:
#   Move top disk from peg src to peg dst
action_to_move = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


class HanoiEnv(gym.Env):
    default_iter = 10
    default_steps = 30

    def __init__(
        self,
        max_steps=5,
        num_disks=4,
        env_noise=0,
        show_valid_actions=False,  # <--- New optional param for showing invalid actions to the observation
        reward_shaping=False,  # <--- New optional param for reward shaping
    ):
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.show_valid_actions = show_valid_actions
        self.reward_shaping = reward_shaping

        # Basic Gym spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple(self.num_disks * (spaces.Discrete(3),))

        # State & Goal
        self.current_state = None
        self.goal_state = self.num_disks * (2,)  # e.g., all disks on peg 2
        self.history = HistoryTracker(max_steps)
        self.done = None
        self.ACTION_LOOKUP = {
            0: "(0,1) - top disk of pole 0 to top of pole 1 ",
            1: "(0,2) - top disk of pole 0 to top of pole 2 ",
            2: "(1,0) - top disk of pole 1 to top of pole 0",
            3: "(1,2) - top disk of pole 1 to top of pole 2",
            4: "(2,0) - top disk of pole 2 to top of pole 0",
            5: "(2,1) - top disk of pole 2 to top of pole 1",
        }

        # Action descriptions
        self.action_list = [
            "Move the top disk of rod A to the top of rod B",
            "Move the top disk of rod A to the top of rod C",
            "Move the top disk of rod B to the top of rod A",
            "Move the top disk of rod B to the top of rod C",
            "Move the top disk of rod C to the top of rod A",
            "Move the top disk of rod C to the top of rod B",
        ]

        self.desc = """
The game consists of three rods (A,B,C) and a number of disks of various sizes, which can go onto any rod. 
The game begins with the disks stacked on rod A in order of decreasing size, the smallest at the top (righthand side). 
The objective is to move the entire stack to rod C, obeying the following rules:

 - Only one disk may be moved at a time.
 - Each move consists of taking the top disk from one of the stacks and placing it on top of another stack or on an empty rod.
 - You cannot place a bigger disk on top of a smaller disk.

For example, considering movements from B under the following setting:
- A: |bottom, [0], top|
- B: |bottom, [1], top|
- C: |bottom, [2], top|
You are only allowed to move from B to C but not A, since the top of B (1) is smaller than the top of C (2) but bigger than the top of A (0).

Finally, the starting configuration is:
- A: |bottom, {}, top|
- B: |bottom, [], top|
- C: |bottom, [], top|

and the goal configuration is:
- A: |bottom, [], top|
- B: |bottom, [], top|
- C: |bottom, {}, top|
with top on the right and bottom on the left

{}
""".format(
            list(range(num_disks))[::-1],
            list(range(num_disks))[::-1],
            describe_act(self.action_list),
        ).strip()

    def step(self, action):
        """
        * Inputs:
            - action: integer from 0 to 5 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: state after transition
            - reward: reward from transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 5 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode finished. Call env.reset() to start a new one.")

        info = {"transition_failure": False, "invalid_action": False}

        # Possibly override chosen action if there's environment noise
        if self.env_noise > 0:
            if random.random() <= self.env_noise:
                action = random.randint(0, self.action_space.n - 1)
                info["transition_failure"] = True
        else:
            info["transition_failure"] = False

        # Check if the action is valid
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            info["invalid_action"] = True

        # We compute reward according to whether it's valid or not
        reward = self.compute_reward(info["invalid_action"])

        # If it's valid, do the move; otherwise we skip
        if not info["invalid_action"]:
            move = action_to_move[action]
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self.current_state)
            moved_state[disk_to_move] = move[1]
            self.current_state = tuple(moved_state)

        # Check completion
        if self.current_state == self.goal_state:
            # Add a large bonus for finishing
            reward += 100
            self.done = True

        # Build info
        info["state"] = (
            self.disks_on_peg(0),
            self.disks_on_peg(1),
            self.disks_on_peg(2),
        )
        info["score"] = len(self.disks_on_peg(2))
        info["manual"] = self.desc
        # Return a textual description that might show valid actions
        info["obs"] = self.describe_state(info, action)

        # Tracking
        info["completed"] = 1 if self.done else 0
        self.history.step(info)

        return self.current_state, reward, self.done, info

    def compute_reward(self, invalid_action):
        """
        Applies either the default or shaped reward scheme.
        """
        if not self.reward_shaping:
            # Original scheme:
            # -1 if invalid
            #  0 if valid
            # +100 if goal completed (applied separately)
            return -1 if invalid_action else 0
        else:
            # Reward Shaping Example:
            # -2 if invalid
            # +1 if valid
            # +100 on goal (applied separately in step)
            if invalid_action:
                return -2
            else:
                return +1

    def get_valid_actions(self):
        """
        In the combined scheme, if `show_valid_actions` is True or False,
        we still compute what is valid. The environment penalizes invalid moves.
        """
        valid = []
        for idx, (src, dst) in enumerate(action_to_move):
            if self.move_allowed((src, dst)):
                valid.append(idx)
        return valid

    def describe_state(self, state, action=None):
        """
        Returns a textual representation of rods and optionally includes valid moves.
        """
        rod_names = ["A", "B", "C"]
        # Show attempted action:
        if action is not None:
            result = f"You tried to {self.action_list[action].lower()}. Current configuration:"
        else:
            result = "Current configuration:"

        # Rod contents
        for i in range(3):
            # Reverse to show top at the end
            result += f"\n- {rod_names[i]}: |bottom, {state['state'][i][::-1]}, top|"

        # If we combined show_valid_actions = True, let's list them in the text
        if self.show_valid_actions:
            valid_actions = self.get_valid_actions()
            if valid_actions:
                # Convert each valid action to a user-friendly string
                valid_strs = [
                    f"{idx+1}. {self.action_list[idx]}" for idx in valid_actions
                ]
                result += f"\n\nValid actions: {', '.join(valid_strs)}"
            else:
                result += "\n\nValid actions: None"

        return result.strip()

    def move_allowed(self, move):
        """
        Checks if we can move top disk from move[0] to move[1].
        1) The source rod must have at least one disk.
        2) We cannot place a bigger disk on top of a smaller disk.
        """
        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])
        if disks_from:
            # If 'to' rod is empty or top disk on 'to' rod is bigger than top disk on 'from'
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def disks_on_peg(self, peg):
        """
        Returns a list of disk IDs on the given peg.
        Smaller ID can mean physically smaller or bigger disk
        depending on your naming scheme, but it’s consistent.
        """
        return [
            disk for disk in range(self.num_disks) if self.current_state[disk] == peg
        ]

    def reset(self):
        """
        Reset all disks to rod A (peg 0).
        """
        self.current_state = self.num_disks * (0,)
        self.done = False
        self.history.reset()

        info = {
            "state": (
                self.disks_on_peg(0),
                self.disks_on_peg(1),
                self.disks_on_peg(2),
            )
        }
        info["score"] = len(self.disks_on_peg(2))
        info["manual"] = self.desc
        info["obs"] = self.describe_state(info)
        info["completed"] = 0
        self.history.step(info)

        return self.current_state, info

    def render(self, mode="human", close=False):
        """Not used in this text-based environment"""
        pass

    def set_env_parameters(self, num_disks=4, env_noise=0, verbose=True):
        """
        Dynamically change the environment parameters if you like.
        """
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.observation_space = spaces.Tuple(self.num_disks * (spaces.Discrete(3),))
        self.goal_state = self.num_disks * (2,)

        if verbose:
            print("Hanoi Environment parameters updated to:")
            print(f" - Disks: {self.num_disks}")
            print(f" - Noise Probability: {self.env_noise}")

    def get_movability_map(self, fill=False):
        """
        Returns a map of valid moves for all possible states, optional usage.
        """
        mov_map = np.zeros(self.num_disks * (3,) + (6,))

        if fill:
            # List out all permutations of disk placements
            id_list = self.num_disks * [0] + self.num_disks * [1] + self.num_disks * [2]
            states = list(itertools.permutations(id_list, self.num_disks))
            for state in states:
                for action in range(6):
                    move = action_to_move[action]
                    disks_from = []
                    disks_to = []
                    for d in range(self.num_disks):
                        if state[d] == move[0]:
                            disks_from.append(d)
                        elif state[d] == move[1]:
                            disks_to.append(d)

                    if disks_from:
                        valid = (min(disks_to) > min(disks_from)) if disks_to else True
                    else:
                        valid = False

                    if not valid:
                        mov_map[state][action] = -np.inf

        return mov_map


# -------------------------------------------------------------------
# Original classes kept for backward compatibility
# (they default to show_valid_actions=False, reward_shaping=False)
# -------------------------------------------------------------------


class Hanoi3Disk(HanoiEnv):
    """Basic 3 disk Hanoi Environment"""

    def __init__(
        self, max_steps=5, env_noise=0, show_valid_actions=False, reward_shaping=False
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
        )


class Hanoi4Disk(HanoiEnv):
    """Basic 4 disk Hanoi Environment"""

    def __init__(
        self, max_steps=5, env_noise=0, show_valid_actions=False, reward_shaping=False
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=4,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
        )


# -------------------------------------------------------------------
# NEW specialized classes with 2disks, show_valid_actions and reward_shaping
# -------------------------------------------------------------------


class Hanoi2Disk(HanoiEnv):
    """Basic 2 disk Hanoi Environment"""

    def __init__(
        self, max_steps=5, env_noise=0, show_valid_actions=False, reward_shaping=False
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=2,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
        )


class Hanoi2DShowValid(HanoiEnv):
    """
    2-disk environment that always shows valid actions and enforces
    invalid action penalty.
    """

    def __init__(self, max_steps=5, env_noise=0):
        super().__init__(
            max_steps=max_steps,
            num_disks=2,
            env_noise=env_noise,
            show_valid_actions=True,
            reward_shaping=False,
        )


class Hanoi2DRewardShaping(HanoiEnv):
    """
    2-disk environment using reward shaping.
    """

    def __init__(self, max_steps=5, env_noise=0):
        super().__init__(
            max_steps=max_steps,
            num_disks=2,
            env_noise=env_noise,
            show_valid_actions=False,
            reward_shaping=True,
        )


class Hanoi2DShowValidRewardShaping(HanoiEnv):
    """
    2-disk environment with valid actions displayed and reward shaping.
    """

    def __init__(self, max_steps=5, env_noise=0):
        super().__init__(
            max_steps=max_steps,
            num_disks=2,
            env_noise=env_noise,
            show_valid_actions=True,
            reward_shaping=True,
        )


class Hanoi3DShowValid(HanoiEnv):
    """
    3-disk environment that always shows valid actions, classic reward scheme.
    """

    def __init__(self, max_steps=5, env_noise=0):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=True,
            reward_shaping=False,
        )


class Hanoi3DRewardShaping(HanoiEnv):
    """
    3-disk environment using reward shaping.
    """

    def __init__(self, max_steps=5, env_noise=0):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=False,
            reward_shaping=True,
        )


class Hanoi3DShowValidRewardShaping(HanoiEnv):
    """
    3-disk environment with valid actions displayed and reward shaping.
    """

    def __init__(self, max_steps=5, env_noise=0):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=True,
            reward_shaping=True,
        )
