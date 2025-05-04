# Benchmark games used in the paper
benchmark_games_v0 = [
    "RockPaperScissorBasic",
    "BanditTwoArmedHighLowFixed",
    "MessengerL1",
    "MessengerL2",
    "Hanoi3Disk",
    "Crafter",
    "MinedojoCreative0",
    # Newly added environments
    "Hanoi2Disk",
    "Hanoi2DShowValid",
    "Hanoi2DRewardShaping",
    "Hanoi2DShowValidRewardShaping",
    "Hanoi3DShowValid",
    "Hanoi3DRewardShaping",
    "Hanoi3DShowValidRewardShaping",
    # Newly added environments
    "MessengerL1Shaped",
    "MessengerL1NoRand",
    "MessengerL1ShapedNoRand",
]


import numpy as np
import pandas as pd

from . import game_challenges, recorded_settings


def normalize_score(game_name, score):
    # Normalize score to [0, 1] for a given environment

    if game_name in recorded_settings.keys():
        return (score - recorded_settings[game_name]["min score"]) / (
            recorded_settings[game_name]["human score"]
            - recorded_settings[game_name]["min score"]
        )
    else:
        raise ValueError(
            "Environment `{}` does not have recorded settings.".format(game_name)
        )


def analyze_capabilities(score_dict):
    # Analyze capabilities of a model based on the metric dictionary.
    # The metric dictionary should have the following format:
    # {
    #     'game_name': score,
    #     ...
    # }
    # The output is a dictionary with the following format:
    # {
    #     'game_name': capability,
    #     ...
    # }

    challenges_df = pd.DataFrame(game_challenges)
    challenges_df = challenges_df.loc[:, list(score_dict.keys())] - 1

    normalized_scores = np.array(list(score_dict.values())).reshape(-1, 1)
    scores = (challenges_df.values @ normalized_scores).squeeze() / np.sum(
        challenges_df.values, axis=1
    )

    return dict(zip(challenges_df.index, scores.flatten()))
