import logging
import os
from typing import Optional, Union

import numpy as np
import wandb


def initialize_wandb_experiment(env_name: str, config: dict):
    if config["use_reflection"] and not config["use_oracle"]:
        name = f"{env_name}_reflection"
    elif (
        config["use_reflection"] and config["use_oracle"] and not config["use_planner"]
    ):
        name = f"{env_name}_reflection_oracle"
    elif config["use_planner"] and config["use_reflection"]:
        name = f"{env_name}_reflection_planner"
    elif (
        not config["use_reflection"]
        and not config["use_oracle"]
        and not config["use_planner"]
    ):
        name = f"{env_name}_default"
    else:
        raise ValueError(
            "Invalid configuration. The allowed combinations are: "
            "(1) Default: use_reflection=False, use_oracle=False, use_planner=False "
            "(2) Reflection only: use_reflection=True, use_oracle=False, use_planner=False "
            "(3) Reflection + Oracle: use_reflection=True, use_oracle=True, use_planner=False "
            "(4) Reflection + Planner: use_reflection=True, use_oracle=False/True, use_planner=True"
        )
    wandb.init(
        project="LLM_Agent",
        name=name,
        config=config,
    )
    log_artifacts()


def log_artifacts(prompts_dir: str = "src/prompts"):
    run = wandb.run

    # Create a new artifact
    artifact = wandb.Artifact(name="prompts_artifact", type="dataset")

    # Add files to the artifact
    for filename in os.listdir(prompts_dir):
        file_path = os.path.join(prompts_dir, filename)
        if os.path.isfile(file_path):
            artifact.add_file(file_path, name=filename)

    # Log the artifact
    run.log_artifact(artifact)


def create_episode_table():
    """
    Create a new WandB table for an episode.
    """
    columns = [
        "step",
        "prompt",
        "observation",
        "score",
        "reward",
        "total_reward",
        "answer",
        "action_index",
        "action",
        "invalid_action",
        "reflection_output",
        "oracle_output",
        "planner_prompt",
        "planner_output",
        "input_tokens",
        "output_tokens",
    ]
    return wandb.Table(columns=columns)


def log_episode_metrics(
    step: int,
    prompt: str,
    observation: str,
    score: float,
    reward: float,
    total_reward: float,
    answer: str,
    action_index: int,
    action: str,
    invalid_action: bool,
    reflection_output: str,
    oracle_output: str,
    planner_prompt: str,
    planner_output: str,
    wandb_table: wandb.Table,
    input_tokens: Optional[Union[int, str]] = None,
    output_tokens: Optional[Union[int, str]] = None,
    model_type: Optional[str] = None,
):
    """
    Log metrics for each step of the episode.
    """

    # Clean up for openai
    if model_type == "openai":
        prompt = prompt[0].get("content")
        if oracle_output:
            oracle_output = (
                oracle_output
                if isinstance(oracle_output, str)
                else oracle_output[0].get("content")
            )

    new_row = [
        step,
        prompt,
        observation,
        score,
        reward,
        total_reward,
        answer,
        action_index,
        action,
        invalid_action,
        reflection_output,
        oracle_output,
        planner_prompt,
        planner_output,
        input_tokens,
        output_tokens,
    ]
    wandb_table.add_data(*new_row)


def plot_rewards_to_wandb(
    rewards: list,
    plot_id: str = "rewards_plot",
    title: str = "Rewards vs Timesteps",
    folder_path: str = None,
):
    """
    Plots a list of rewards to Weights and Biases (wandb) in a specified folder structure.

    Args:
        rewards (list): A list of rewards to plot.
        eps (int): The episode number.
        plot_id (str): The identifier for the plot in wandb.
        title (str): The title of the plot.
    """
    # Create x-values (e.g., timesteps or indices)
    x_values = list(range(len(rewards)))

    # Pair x-values with rewards
    data = [[x, y] for (x, y) in zip(x_values, rewards)]

    # Create a wandb.Table
    table = wandb.Table(data=data, columns=["x", "y"])

    # Determine the folder path
    folder_path = folder_path if folder_path else ""

    # Log the line plot to wandb
    wandb.log(
        {f"{folder_path}{plot_id}": wandb.plot.line(table, "x", "y", title=title)}
    )


def finalize_episode_logging(
    episode_number: int, wandb_table: wandb.Table, rewards: list
):
    """
    Finalize logging for an episode, logging the table and summary graphs.
    """
    logging.info(f"Logging episode {episode_number} to WandB")

    wandb.log({f"episode_{episode_number}/rollout_table": wandb_table})

    plot_rewards_to_wandb(rewards, episode_number)


def finalize_experiment_logging(
    normalized_scores, rewards, best_rewards_so_far=None, best_so_far=None
):
    """
    Finalizes the WandB run by plotting various metrics and logging the best score.

    Args:
        normalized_scores (list or np.array): Normalized scores per episode.
        rewards (list or np.array): Rewards per episode.
        best_rewards_so_far (list or np.array, optional): Best rewards so far over episodes.
        best_so_far (tuple, optional): Tuple containing the best score information.
    """
    # Plot normalized scores
    plot_rewards_to_wandb(
        normalized_scores,
        plot_id="overall_normalized_scores",
        title="Normalized score per episode",
        folder_path="summary/",
    )

    # Plot cumulative rewards
    plot_rewards_to_wandb(
        np.cumsum(rewards),
        plot_id="cumulative_rewards",
        title="Cumulative rewards over episodes",
        folder_path="summary/",
    )

    # Plot rewards per episode
    plot_rewards_to_wandb(
        rewards,
        plot_id="episode_rewards",
        title="Rewards over episodes",
        folder_path="summary/",
    )

    # Plot best rewards so far if available
    if best_rewards_so_far:
        plot_rewards_to_wandb(
            best_rewards_so_far,
            plot_id="best_rewards_so_far",
            title="Best rewards so far over episodes",
            folder_path="summary/",
        )

    # Log best score so far if available
    if best_so_far:
        wandb.log({"summary/best_final_score": best_so_far[1]})

        table = wandb.Table(columns=["Heuristics"])
        table.add_data(str(best_so_far[0]))

        # Log the table to WandB
        wandb.log({"summary/best_final_heuristics": table})

    # Finalize experiment logging
    wandb.finish()
