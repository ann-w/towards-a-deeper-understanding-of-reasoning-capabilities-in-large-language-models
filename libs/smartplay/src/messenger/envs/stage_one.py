"""
Classes that follow a gym-like interface and implement stage one of the Messenger environment.
"""

import json
import random
from collections import namedtuple
from pathlib import Path

import messenger.envs.config as config
import numpy as np
from messenger.envs.base import MessengerEnv, Position
from messenger.envs.manual import TextManual
from messenger.envs.utils import games_from_json

# Used to track sprites in StageOne, where we do not use VGDL to handle sprites.
Sprite = namedtuple("Sprite", ["name", "id", "position"])


class StageOne(MessengerEnv):
    def __init__(
        self,
        split: str,
        message_prob: float = 0.2,
        shuffle_obs: bool = True,
        use_shaping: bool = False,
        use_text_substitution: bool = False,
    ):
        """
        Stage one where objects are all immovable. Since the episode length is short and entities
        do not move, we do not use the VGDL engine for efficiency.

        split:
            Which dataset split (train/val/test) to load.
        message_prob:
            The probability that the avatar starts holding the message (WITH_MESSAGE).
        shuffle_obs:
            If True, shuffle the textual descriptions in the observation.
        use_shaping:
            If True, enable distance-based reward shaping. If False, do not use it.
        use_text_substitution:
            If True, use random text descriptions for entities. If False, use raw entity names.
        """
        super().__init__(lvl=1)
        self.message_prob = message_prob
        self.shuffle_obs = shuffle_obs
        self.use_shaping = use_shaping
        self.use_text_substitution = use_text_substitution
        this_folder = Path(__file__).parent

        # Get the games and manual
        games_json_path = this_folder.joinpath("games.json")
        if "train" in split and "mc" in split:  # multi-combination games
            game_split = "train_multi_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "train" in split and "sc" in split:  # single-combination games
            game_split = "train_single_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "val" in split:
            game_split = "val"
            text_json_path = this_folder.joinpath("texts", "text_val.json")
        elif "test" in split:
            game_split = "test"
            text_json_path = this_folder.joinpath("texts", "text_test.json")
        else:
            raise Exception(f"Split: {split} not understood.")

        # list of Game namedtuples
        self.all_games = games_from_json(json_path=games_json_path, split=game_split)

        # we only need the immovable and unknown descriptions, so just extract those.
        with text_json_path.open(mode="r") as f:
            descrip = json.load(f)

        self.descriptors = {}
        for entity in descrip:
            self.descriptors[entity] = {}
            for role in ("enemy", "message", "goal"):
                self.descriptors[entity][role] = []
                for sent in descrip[entity][role]["immovable"]:
                    self.descriptors[entity][role].append(sent)
                for sent in descrip[entity][role]["unknown"]:
                    self.descriptors[entity][role].append(sent)

        # all possible entity locations
        self.positions = [
            Position(y=3, x=5),
            Position(y=5, x=3),
            Position(y=5, x=7),
            Position(y=7, x=5),
        ]
        self.avatar_start_pos = Position(y=5, x=5)
        self.avatar = None
        self.enemy = None
        self.message = None
        self.goal = None

        # For shaping:
        self.prev_dist_to_message = None
        self.prev_dist_to_goal = None

    def _get_manual(self):
        if self.use_text_substitution:
            # Existing behavior with random descriptions
            enemy_str = random.choice(self.descriptors[self.enemy.name]["enemy"])
            key_str = random.choice(self.descriptors[self.message.name]["message"])
            goal_str = random.choice(self.descriptors[self.goal.name]["goal"])
        else:
            # Direct entity names without substitution
            enemy_str = f"Avoid the {self.enemy.name}"
            key_str = f"Pick up the {self.message.name}"
            goal_str = f"Deliver to the {self.goal.name}"

        manual = [enemy_str, key_str, goal_str]
        if self.shuffle_obs:
            random.shuffle(manual)
        return manual

    def _get_obs(self):
        entities = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        avatar = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        for sprite in (self.enemy, self.message, self.goal):
            entities[sprite.position.y, sprite.position.x, 0] = sprite.id

        avatar[self.avatar.position.y, self.avatar.position.x, 0] = self.avatar.id

        return {"entities": entities, "avatar": avatar}

    def _manhattan_distance(self, sprite1, sprite2):
        """Helper for integer grid distance."""
        manhattan_distance = abs(sprite1.position.x - sprite2.position.x) + abs(
            sprite1.position.y - sprite2.position.y
        )
        return manhattan_distance

    def reset(self):
        self.game = random.choice(self.all_games)
        enemy, message, goal = self.game.enemy, self.game.message, self.game.goal

        # randomly choose where to put enemy, message, goal
        shuffled_pos = random.sample(self.positions, 4)
        self.enemy = Sprite(name=enemy.name, id=enemy.id, position=shuffled_pos[0])
        self.message = Sprite(
            name=message.name, id=message.id, position=shuffled_pos[1]
        )
        self.goal = Sprite(name=goal.name, id=goal.id, position=shuffled_pos[2])

        # Decide whether avatar starts with or without the message
        if random.random() < self.message_prob:
            self.avatar = Sprite(
                name=config.WITH_MESSAGE.name,
                id=config.WITH_MESSAGE.id,
                position=self.avatar_start_pos,
            )
        else:
            self.avatar = Sprite(
                name=config.NO_MESSAGE.name,
                id=config.NO_MESSAGE.id,
                position=self.avatar_start_pos,
            )

        # Initialize distances for shaping if needed
        self.prev_dist_to_message = self._manhattan_distance(self.avatar, self.message)
        self.prev_dist_to_goal = self._manhattan_distance(self.avatar, self.goal)

        obs = self._get_obs()
        manual = self._get_manual()

        return obs, manual

    def _move_avatar(self, action):
        """
        Updates the agent's position based on the selected action.

        The agent moves within the game grid, avoiding out-of-bounds movements.
        Possible actions:
        - Stay in place
        - Move Up
        - Move Down
        - Move Left
        - Move Right
        """
        # print(f"Before move: {self.avatar.position}, Action: {action}")  # Debugging output

        # Action: Stay in place (No movement)
        if action == config.ACTIONS.stay:
            # print("Action: Stay → No movement")
            return

        elif action == config.ACTIONS.up:
            if self.avatar.position.y <= 0:  # top boundary
                # print("Hit upper boundary. No movement.")
                return
            new_position = Position(
                y=self.avatar.position.y - 1, x=self.avatar.position.x
            )

        elif action == config.ACTIONS.down:
            if self.avatar.position.y >= config.STATE_HEIGHT - 1:  # bottom boundary
                # print("Hit lower boundary. No movement.")
                return
            new_position = Position(
                y=self.avatar.position.y + 1, x=self.avatar.position.x
            )

        elif action == config.ACTIONS.left:
            if self.avatar.position.x <= 0:  # left boundary
                # print("Hit left boundary. No movement.")
                return
            new_position = Position(
                y=self.avatar.position.y, x=self.avatar.position.x - 1
            )

        elif action == config.ACTIONS.right:
            if self.avatar.position.x >= config.STATE_WIDTH - 1:  # right boundary
                # print("Hit right boundary. No movement.")
                return
            new_position = Position(
                y=self.avatar.position.y, x=self.avatar.position.x + 1
            )

        else:
            raise Exception(f"{action} is not a valid action.")

        # Update the avatar's position
        self.avatar = Sprite(
            name=self.avatar.name, id=self.avatar.id, position=new_position
        )
        # print(f"After move: {self.avatar.position}")  # Debugging output

    def _overlap(self, sprite_1, sprite_2):
        overlap = (
            sprite_1.position.x == sprite_2.position.x
            and sprite_1.position.y == sprite_2.position.y
        )
        # if overlap:
        # print(f"Overlap detected: {sprite_1.name} and {sprite_2.name}")
        return overlap

    def _has_message(self):
        """Return True if the avatar currently has the message."""
        return self.avatar.name == config.WITH_MESSAGE.name

    def step(self, action):
        # print(f"Before move: Avatar at {self.avatar.position}, distance to message: {self.prev_dist_to_message}, distance to goal: {self.prev_dist_to_goal}")

        # 1) Move the avatar
        self._move_avatar(action)
        obs = self._get_obs()

        # Compute distances to message and goal
        new_dist_to_message = self._manhattan_distance(self.avatar, self.message)
        new_dist_to_goal = self._manhattan_distance(self.avatar, self.goal)
        # print(f"After move: Avatar at {self.avatar.position}, distance to message: {new_dist_to_message}, distance to goal: {new_dist_to_goal}")

        # 2) Optional: reward shaping
        shaping_reward = 0.0
        if self.use_shaping:
            # print("Reward shaping enabled.")

            if not self._has_message():
                # Shaping is about getting closer to the message
                delta_msg = self.prev_dist_to_message - new_dist_to_message
                # Clamp movement difference to avoid large jumps
                delta_msg = max(-1, min(1, delta_msg))
                shaping_reward += (
                    0.5 * delta_msg
                )  # small reward for getting closer to message without overwhelming main game rewards

            else:
                # Once we have the message, shape toward the goal
                delta_goal = self.prev_dist_to_goal - new_dist_to_goal
                delta_goal = max(-1, min(1, delta_goal))
                shaping_reward += 0.5 * delta_goal

            # Update previous distances
            self.prev_dist_to_message = new_dist_to_message
            self.prev_dist_to_goal = new_dist_to_goal

        # 3) Check collisions and compute final reward
        final_reward = 0.0
        done = False

        # Overlap with enemy => immediate negative
        if self._overlap(self.avatar, self.enemy):
            final_reward = -1.0
            done = True

        # Overlap with message => pick up or fail
        elif self._overlap(self.avatar, self.message):
            done = True
            if self._has_message():
                # Already has message => negative reward
                final_reward = -1.0
            else:
                # Transition from NO_MESSAGE to WITH_MESSAGE
                # if reward shaping is enabled
                if self.use_shaping:
                    final_reward = 10.0
                else:
                    final_reward = 1.0

        # Overlap with goal => success or fail
        elif self._overlap(self.avatar, self.goal):
            done = True
            if self._has_message():
                if self.use_shaping:
                    # Big reward for success
                    final_reward = 50.0
                else:
                    final_reward = 1.0
            else:
                final_reward = -1.0

        # 4) Combine shaping + final
        total_reward = shaping_reward + final_reward

        return obs, total_reward, done, {}
