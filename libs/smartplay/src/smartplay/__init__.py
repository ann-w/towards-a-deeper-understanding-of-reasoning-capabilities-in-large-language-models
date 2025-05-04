# This file is used to load all games in the smartplay directory.

import importlib
import os
import warnings

import yaml

root_dir = os.path.dirname(os.path.abspath(__file__))
games = []
env_list = []


_exclude_path = ["__pycache__", "utils", "tests", "eval", "example_game"]
_module_dir = os.path.dirname(__file__)

for dirname in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, dirname)) and dirname not in _exclude_path:
        if "__init__.py" in os.listdir(os.path.join(root_dir, dirname)):
            games.append(dirname)


game_challenges = {}
recorded_settings = {}


for game in games:

    if not os.path.exists(os.path.join(root_dir, game, "evaluation.yml")):
        warnings.warn(
            "Game `{}` does not have evaluation.yml. Skipping the game.".format(game),
            UserWarning,
        )
        continue

    try:
        # Load evaluation settings
        with open(os.path.join(root_dir, game, "evaluation.yml"), "r") as f:
            yml = yaml.safe_load(f)
            recorded_setting = yml["recorded settings"]
            game_challenge = yml["challenges"]

    except Exception as e:
        warnings.warn(
            "Failed to load `evaluation.yml` for `{}`.".format(game), UserWarning
        )
        continue

    try:
        # Load environments
        module = importlib.import_module("." + game, package="smartplay")

        environments = getattr(module, "environments", None)

        if environments:
            for env_def in environments:
                if len(env_def) == 2:
                    env_name, version = env_def
                    kwargs = {}
                else:
                    env_name, version, kwargs = env_def

                env_list.append(f"{env_name}-{version}")
                game_challenges[env_name] = (
                    game_challenge["all"]
                    if env_name not in game_challenge
                    else game_challenge[env_name]
                )
                if env_name in recorded_setting:
                    recorded_settings[env_name] = recorded_setting[env_name]
        else:
            print(f"  No `environments` found in {game} or it's None")  # Debug
            warnings.warn(
                f"Failed to load `{game}.environments`. Skipping the game.", UserWarning
            )
            continue

    except Exception as e:
        warnings.warn(
            "Failed to import `{}`. Skipping the game.".format(game), UserWarning
        )
        continue


from .eval import *
