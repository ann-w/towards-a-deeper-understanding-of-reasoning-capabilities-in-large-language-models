import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def initialize_results_dicts() -> Tuple[Dict, Dict]:
    """
    Initialize the results and score dictionaries.

    Returns:
    Tuple[Dict, Dict]: The initialized results and score dictionaries.
    """
    results_dict = {}
    score_dict = {}

    return results_dict, score_dict


def update_results(
    env_name: str,
    normalized_scores: List,
    rewards: List,
    scores: List,
    results_dict: Dict,
    score_dict: Dict,
) -> None:
    """
    Update the results and score dictionaries with the given environment data.

    Parameters:
    env_name (str): The name of the environment.
    normalized_scores (list): The list of normalized scores.
    rewards (list): The list of rewards.
    scores (list): The list of scores.
    results_dict (dict): The dictionary to store detailed results.
    score_dict (dict): The dictionary to store average normalized scores.
    """
    results_dict[env_name] = {
        "normalized_scores": normalized_scores,
        "rewards": rewards,
        "scores": scores,
    }
    score_dict[env_name] = sum(normalized_scores) / len(normalized_scores)


def calculate_statistics(results_dict: Dict) -> Dict:
    """
    Calculate the mean and standard deviation of the scores, rewards, and normalized scores from the results dictionary.

    Parameters:
    results_dict (dict): The dictionary containing detailed results for each environment.

    Returns:
    dict: A dictionary containing the mean and standard deviation of scores, rewards, and normalized scores for each environment.
    """
    statistics = {}
    for env_name, env_data in results_dict.items():
        total_normalized_scores = env_data["normalized_scores"]
        total_rewards = env_data["rewards"]
        total_scores = env_data["scores"]

        mean_normalized_scores = np.mean(total_normalized_scores)
        std_normalized_scores = np.std(total_normalized_scores)
        mean_rewards = np.mean(total_rewards)
        std_rewards = np.std(total_rewards)
        mean_scores = np.mean(total_scores)
        std_scores = np.std(total_scores)

        statistics[env_name] = {
            "mean_normalized_scores": float(round(mean_normalized_scores, 3)),
            "std_normalized_scores": float(round(std_normalized_scores, 3)),
            "mean_rewards": float(round(mean_rewards, 3)),
            "std_rewards": float(round(std_rewards, 3)),
            "mean_scores": float(round(mean_scores, 3)),
            "std_scores": float(round(std_scores, 3)),
        }

    return statistics


def plot_results(results_dict: Dict, output_folder: str = "experiment_logs") -> None:
    """
    Plot the normalized scores, rewards, and scores for each environment,
    including shaded areas for standard deviation.

    Parameters:
    results_dict (dict): The dictionary containing detailed results for each environment.
    """
    # get date and time for the plot
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # if output folder not exists, create it
    output_folder = f"{output_folder}/{dt_string}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for env_name, data in results_dict.items():
        normalized_scores = np.array(data["normalized_scores"])
        rewards = np.array(
            data["rewards"], dtype=np.float64
        )  # Convert rewards to float for calculations
        scores = np.array(data["scores"])

        episodes = range(len(normalized_scores))

        # Calculate mean and standard deviation
        norm_mean, norm_std = np.mean(normalized_scores), np.std(normalized_scores)
        reward_mean, reward_std = np.mean(rewards), np.std(rewards)
        score_mean, score_std = np.mean(scores), np.std(scores)

        # Plot normalized scores
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(normalized_scores, marker="o", label="Normalized Scores")
        plt.fill_between(
            range(len(normalized_scores)),
            normalized_scores - norm_std,
            normalized_scores + norm_std,
            color="blue",
            alpha=0.2,
            label="Std Dev",
        )
        plt.title(f"Normalized Scores - {env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Normalized Score")
        plt.xticks(episodes)  # Set x-axis ticks to be integers
        plt.legend()

        # Plot rewards
        plt.subplot(1, 3, 2)
        plt.plot(rewards, marker="o", label="Rewards")
        plt.fill_between(
            range(len(rewards)),
            rewards - reward_std,
            rewards + reward_std,
            color="orange",
            alpha=0.2,
            label="Std Dev",
        )
        plt.title(f"Rewards - {env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.xticks(episodes)  # Set x-axis ticks to be integers
        plt.legend()

        # Plot scores
        plt.subplot(1, 3, 3)
        plt.plot(scores, marker="o", label="Scores")
        plt.fill_between(
            range(len(scores)),
            scores - score_std,
            scores + score_std,
            color="green",
            alpha=0.2,
            label="Std Dev",
        )
        plt.title(f"Scores - {env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.xticks(episodes)  # Set x-axis ticks to be integers
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"{output_folder}/{env_name}_results.png"
        )  # Save the plot to a file
        # plt.show(block=True)  # Show the plot
