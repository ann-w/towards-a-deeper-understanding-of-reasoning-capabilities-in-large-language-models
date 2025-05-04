import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from scripts.episode_runner import run_single_episode
from scripts.generate_offspring import (ask_oracle,
                                        generate_offspring_in_parallel,
                                        update_after_generation)
from scripts.memory import Memory
from scripts.oracle_evaluation import evaluate_oracle_offspring
from scripts.utils.environment_utils import initialize_environment

# Set up logging
logging.basicConfig(level=logging.INFO)


def run_single_episode_with_logging(
    env,
    env_name,
    env_steps,
    query_model,
    model_type,
    global_step,
    iteration,
    memory,
    use_reflection,
    reflection_start_step,
    reflection_interval_steps,
    count_token_usage,
    use_planner,
):
    normalized_score, episode_rewards, score = run_single_episode(
        env=env,
        env_name=env_name,
        env_steps=env_steps,
        query_model=query_model,
        model_type=model_type,
        global_step=global_step,
        episode_number=iteration,
        memory=memory,
        oracle_output=None,
        use_reflection=use_reflection,
        reflection_start_step=reflection_start_step,
        reflection_interval_steps=reflection_interval_steps,
        count_token_usage=count_token_usage,
        use_planner=use_planner,
    )
    return normalized_score, episode_rewards, score


def run_agent_on_environment(
    env_name: str,
    query_model: Callable,
    env_steps: int = None,
    num_iter: int = None,
    num_iter_eval: int = 2,  # Number of episodes to evaluate the Oracle outputs
    model_type: str = "huggingface",
    memory_capacity: int = 10000,
    use_reflection: bool = False,
    reflection_start_step: int = 10,
    reflection_interval_steps: int = 1,
    use_oracle: bool = False,
    use_planner: bool = False,
    parents: List[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    num_parents: int = 1,
    max_workers: int = 1,
    num_offspring: int = 4,
    count_token_usage: bool = True,
) -> Tuple[float, List[float], List[str]]:

    # Initialize variables
    normalized_scores = []
    rewards = []
    scores = []

    parents = None
    best_so_far = None
    best_scores_so_far = []
    global_step = 0

    # Initialize the environment
    env, env_steps, num_iter = initialize_environment(env_name, env_steps, num_iter)

    for iteration in tqdm(range(num_iter), desc=f"Running {env_name}"):
        memory = Memory(capacity=memory_capacity)

        if iteration == 0 and use_oracle:
            logging.info(
                "Running the first iteration with reflection only to gather initial reflections."
            )
            normalized_score, episode_rewards, score = run_single_episode_with_logging(
                env,
                env_name,
                env_steps,
                query_model,
                model_type,
                global_step,
                iteration,
                memory,
                use_reflection,
                reflection_start_step,
                reflection_interval_steps,
                count_token_usage,
                use_planner,
            )
            normalized_scores.append(normalized_score)
            rewards.append(episode_rewards)
            scores.append(score)
            global_step += env_steps
            continue

        if use_oracle:
            if not parents:
                oracle_offspring = ask_oracle(
                    env_name=env_name,
                    memory=memory,
                    n_reflections=reflection_start_step,
                    model_type=model_type,
                    query_model=query_model,
                    num_oracle_output=num_offspring,
                )
            else:
                oracle_offspring = generate_offspring_in_parallel(
                    game_description=env.env.desc,
                    model_type=model_type,
                    parents=parents,
                    new_reflections=memory.get_reflections(n=reflection_start_step),
                    query_model=query_model,
                    max_workers=max_workers,
                    num_offspring=num_offspring,
                )

            oracle_offspring_performance = evaluate_oracle_offspring(
                env=env,
                env_name=env_name,
                env_steps=env_steps,
                query_model=query_model,
                model_type=model_type,
                num_iter=num_iter_eval,
                global_step=global_step,
                oracle_offspring=oracle_offspring,
                memory_capacity=memory_capacity,
                use_reflection=use_reflection,
                reflection_start_step=reflection_start_step,
                reflection_interval_steps=reflection_interval_steps,
                max_workers=max_workers,
                episode_num=iteration,
                count_token_usage=count_token_usage,
                use_planner=use_planner,
            )

            logging.info(f"iteration before evaluation: {iteration}")

            parents, normalized_scores, rewards, best_so_far, best_scores_so_far = (
                update_after_generation(
                    oracle_offspring_performance,
                    parents,
                    iteration,
                    normalized_scores,
                    rewards,
                    best_so_far,
                    best_scores_so_far,
                    num_parents=num_parents,
                )
            )

            iteration += num_iter_eval * num_offspring
            logging.info(f"Iteration after evaluation: {iteration} ")
            global_step += env_steps * num_iter_eval * num_offspring
            continue

        logging.info(
            f"Running iteration {iteration} with reflection only."
            if use_reflection
            else f"Running episode {iteration} without reflection or Oracle."
        )
        normalized_score, episode_rewards, score = run_single_episode_with_logging(
            env,
            env_name,
            env_steps,
            query_model,
            model_type,
            global_step,
            iteration,
            memory,
            use_reflection,
            reflection_start_step,
            reflection_interval_steps,
            count_token_usage,
            use_planner,
        )
        normalized_scores.append(normalized_score)
        rewards.append(episode_rewards)
        scores.append(score)
        global_step += env_steps

    return normalized_scores, rewards, scores, best_scores_so_far, best_so_far
