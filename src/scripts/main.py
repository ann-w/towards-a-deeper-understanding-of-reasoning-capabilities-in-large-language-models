import argparse
import logging
import os

import numpy as np
import smartplay
from dotenv import load_dotenv

from scripts.agent_runner import run_agent_on_environment
from scripts.utils.environment_utils import initialize_model
from scripts.utils.wandb_logging import (finalize_experiment_logging,
                                         initialize_wandb_experiment)
from scripts.utils.yaml_utils import (load_experiment_settings,
                                      log_experiment_settings)
from src.scripts.utils.summarize_results import (calculate_statistics,
                                                 initialize_results_dicts,
                                                 plot_results, update_results)

"""This script is the entry point for running the agent on the environment. 
It loads the experiment settings from a YAML file, parses the command-line arguments, and runs the agent on the environment. 
The script also initializes the logging and the Weights and Biases (WandB) experiment."""


def parse_experiment_settings(experiment_settings: dict) -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=experiment_settings.get(
            "model_name", "microsoft/Phi-3-mini-4k-instruct"
        ),
        help="Name of the LLM",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=experiment_settings.get("api_type", "huggingface"),
        choices=[
            "huggingface",
            "local_huggingface",
            "openai",
            "ollama",
            "deepseek",
            "gemini",
        ],
        help="Type of the LLM API",
    )
    parser.add_argument(
        "--env_names",
        type=str,
        default=experiment_settings.get("env_names", "MessengerL1"),
        choices=[
            "MessengerL1",
            "MessengerL2",
            "MessengerL3",
            "RockPaperScissorBasic",
            "RockPaperScissorDifferentScore",
            "BanditTwoArmedDeterministicFixed",
            "BanditTwoArmedHighHighFixed",
            "BanditTwoArmedHighLowFixed",
            "BanditTwoArmedLowLowFixed",
            "Hanoi3Disk",
            "Hanoi4Disk",
            "MinedojoCreative0",
            "MinedojoCreative1",
            "MinedojoCreative2",
            "MinedojoCreative4",
            "MinedojoCreative5",
            "MinedojoCreative7",
            "MinedojoCreative8",
            "MinedojoCreative9",
            "Crafter",
        ],
        help=("Comma separated list of environments to run"),
    )
    parser.add_argument(
        "--env_steps",
        type=int,
        default=experiment_settings.get("env_steps", None),
        help="Number of steps to run the environment for. If not provided, the default number of steps for the environment will be used.",
    )
    parser.add_argument(
        "--env_iter_eval",
        type=int,
        default=experiment_settings.get("env_iter_eval", 2),
        help="Number of iterations/episodes to evaluate the Oracle outputs",
    )
    parser.add_argument(
        "--env_iter",
        type=int,
        default=experiment_settings.get("env_iter", None),
        help="Number of iterations/episodes to run the environment for. If not provided, the default number of iterations for the environment will be used.",
    )
    parser.add_argument(
        "--use_reflection",
        action="store_true",
        default=experiment_settings.get("use_reflection", False),
        help="Whether to use the reflection to assist the agent",
    )
    parser.add_argument(
        "--reflection_start_step",
        type=int,
        default=experiment_settings.get("reflection_start_step", 10),
        help="Step number after which the reflection should start assisting",
    )
    parser.add_argument(
        "--reflection_interval_steps",
        default=experiment_settings.get("reflection_interval_steps", 1),
        type=int,
        help="Number of steps between each reflection",
    )
    parser.add_argument(
        "--use_oracle",
        action="store_true",
        default=experiment_settings.get("use_oracle", False),
        help="Whether to use the Oracle to assist the agent",
    )
    parser.add_argument(
        "--use_planner",
        action="store_true",
        default=experiment_settings.get("use_planner", False),
        help="Whether to use the Planner to assist the agent",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=experiment_settings.get("max_workers", 4),
        help="Maximum number of workers for parallel execution",
    )
    parser.add_argument(
        "--num_offspring",
        default=experiment_settings.get("num_offspring", 4),
        type=int,
        help="Number of offspring to generate in each generation",
    ),
    parser.add_argument(
        "--count_token_usage",
        default=experiment_settings.get("count_token_usage", True),
        action="store_true",  # by default False
        help="Count number of tokens used in each query",
    )
    parser.add_argument(
        "--num_parents",
        default=experiment_settings.get("num_parents", 1),
        type=int,
        help="The number of parents used for generating the offspring",
    )
    return parser.parse_args()


def set_default_values(args):
    if args.env_names is None:
        args.env_names = ",".join(smartplay.benchmark_games_v0)
    return args


def main():
    load_dotenv()

    # Set the working directory to the parent folder that is two levels up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    os.chdir(parent_dir)

    # Load experiment settings
    experiment_settings = load_experiment_settings(
        "src/config/experiment_settings/experiment_settings.yaml"
    )

    logging.info("Loaded Experiment Settings:")
    log_experiment_settings(experiment_settings)

    # Parse arguments again with the loaded experiment settings
    args = parse_experiment_settings(experiment_settings)
    args = set_default_values(args)

    # initialize dicts
    results_dict, score_dict = initialize_results_dicts()

    # initialize model
    query_model = initialize_model(args.model_name, args.model_type)

    for env_name in args.env_names.split(","):

        # Initialize WandB
        initialize_wandb_experiment(env_name=env_name, config=experiment_settings)

        normalized_scores, rewards, scores, best_scores_so_far, best_so_far = (
            run_agent_on_environment(
                env_name=env_name,
                query_model=query_model,
                env_steps=args.env_steps,
                num_iter=args.env_iter,
                num_iter_eval=args.env_iter_eval,
                model_type=args.model_type,
                use_reflection=args.use_reflection,
                reflection_start_step=args.reflection_start_step,
                reflection_interval_steps=args.reflection_interval_steps,
                parents=None,
                num_parents=args.num_parents,
                use_oracle=args.use_oracle,
                max_workers=args.max_workers,
                num_offspring=args.num_offspring,
                count_token_usage=args.count_token_usage,
                use_planner=args.use_planner,
            )
        )

        logging.info(
            f"[COMPLETED ENVIRONMENT EPISODES] env: {env_name}, average scores: mean: {np.mean(scores)} and std: {np.std(scores)}"
        )

        update_results(
            env_name, normalized_scores, rewards, scores, results_dict, score_dict
        )

        finalize_experiment_logging(
            normalized_scores, rewards, best_scores_so_far, best_so_far
        )

    # Evaluate the capability scores of the LLM based on a given dictionary of scores for different environments
    # It returns a dictionary of capability scores for each environment
    # logging.info(f"Capability scores of the LLM:", smartplay.analyze_capabilities(score_dict))

    statistics = calculate_statistics(results_dict)
    for env_name, stats in statistics.items():
        logging.info(f"Statistics for {env_name}: {stats}")

    plot_results(results_dict, output_folder=f"experiments_logs/{args.model_name}/")


if __name__ == "__main__":
    main()
