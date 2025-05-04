import concurrent.futures
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import gym
import numpy as np

from scripts.compose_ingame_prompt import (get_oracle_output,
                                           load_prompt_template)
from scripts.memory import Memory


def generate_offspring(
    game_description: str,
    model_type: str,
    parent_1: Dict[str, Any],
    new_reflections: str,
    query_model: Callable,
    parent_2: Dict[str, Any] = None,
) -> str:
    """
    Generates offspring oracle_output by combining and mutating the best-performing parent rule sets,
    guided by the agent's most recent reflections.
    """
    # Load the template
    if parent_1 and parent_2:
        template_path: str = "src/prompts/generate_offspring_two_parents.txt"
    elif parent_1:
        template_path: str = "src/prompts/generate_offspring_single_parent.txt"
    else:
        raise ValueError("At least one parent must be provided to generate offspring.")

    prompt_template = load_prompt_template(template_path)

    # Prepare the format dictionary
    format_dict = {
        "parent_heuristics_1": parent_1[0] if isinstance(parent_1, tuple) else parent_1,
        "parent_heuristics_2": (
            parent_2[0] if isinstance(parent_2, tuple) else parent_2 if parent_2 else ""
        ),
        "new_reflections": new_reflections,
        "manual": game_description,
    }

    # Render the prompt using the format dictionary
    offspring_prompt = prompt_template.format(**format_dict)

    if model_type == "openai":
        offspring_prompt = [{"role": "system", "content": offspring_prompt}]

    # Query the model with the generated prompt
    offspring_oracle_output = query_model(offspring_prompt)

    return offspring_oracle_output


def generate_offspring_in_parallel(
    game_description: str,
    model_type: str,
    parents: List[Dict[str, Any]],
    new_reflections: str,
    query_model: Callable,
    max_workers: int = 4,
    num_offspring: int = 4,
    plus_strategy: bool = True,
) -> List[str]:
    """
    Generates offspring oracle_output in parallel by combining and mutating the best-performing parent rule sets,
    guided by the agent's most recent reflections.
    """
    logging.info("Generating offspring in parallel.")

    if len(parents) == 2 and plus_strategy:
        # The next generation has both parents and offspring
        num_offspring = num_offspring - len(parents)

    def generate_single_offspring_wrapper(
        parent_1: Dict[str, Any], parent_2: Dict[str, Any] = None, model_type=model_type
    ) -> str:
        """
        Wrapper to generate offspring for a single pair of parent rule sets.

        Args:
            parent_1 (Dict[str, Any]): The first parent rule set.
            parent_2 (Dict[str, Any]): The second parent rule set (optional).

        Returns:
            Dict[str, Any]: The generated offspring oracle_output as a dictionary.
        """
        return generate_offspring(
            game_description=game_description,
            model_type=model_type,
            parent_1=parent_1,
            parent_2=parent_2,
            new_reflections=new_reflections,
            query_model=query_model,
        )

    # Use ThreadPoolExecutor to parallelize the offspring generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                generate_single_offspring_wrapper,
                parents[0],
                parents[1] if len(parents) == 2 else None,
            )
            for _ in range(num_offspring)
        ]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def ask_oracle(
    env_name: str,
    memory: Memory,
    n_reflections: int,
    model_type: str,
    query_model: Callable,
    num_oracle_output: int,
) -> List[str]:
    """
    Generates a specified number of heuristics sets based on reflections from memory.

    Args:
        env_name (str): Name of the environment.
        memory (Memory): Memory object containing reflections.
        n_reflections (int): Number of reflections to use.
        model_type (str): Type of the model used.
        query_model (Callable): Function to query the model.
        num_oracle_output (int): Number of distinct Oracle rule sets to generate.

    Returns:
        List[str]]: List of distinct Oracle rule sets.
    """
    logging.info("Asking Oracle to generate heuristics from reflections")

    # Get env and info for manual description
    env = gym.make(f"smartplay:{env_name}-v0")
    env.reset()
    _, _, _, info = env.step(0)

    oracle_output = []

    for i in range(num_oracle_output):
        oracle_rule = get_oracle_output(
            info=info,
            memory=memory,
            n_reflections=n_reflections,
            model_type=model_type,
            query_model=query_model,
        )
        oracle_output.append(oracle_rule)

    return oracle_output


def update_after_generation(
    oracle_offspring_performance: List[Tuple[Dict[str, Any], float, int, int]],
    parents: Union[List[Tuple[str, float, int]], None],
    episode: int,
    normalized_scores: List[float],
    rewards: List[float],
    best_so_far: Tuple[str, float, int],
    best_scores_so_far: List[float],
    num_parents: int,
) -> Tuple[
    List[Tuple[Dict[str, Any], float, int]],
    List[float],
    List[float],
    Tuple[str, float, int],
    List[float],
]:

    if not best_scores_so_far:
        best_scores_so_far.append(max(normalized_scores))

    # After the first iteration, the parents are part of the population to evaluate
    if episode > 1 and parents:
        oracle_offspring_performance.extend(parents)

    # Sort parents and get new parents
    sorted_oracle_offspring_performance = sorted(
        oracle_offspring_performance, key=lambda x: x[1], reverse=True
    )
    parents = sorted_oracle_offspring_performance[:num_parents]

    # Save the best parent
    best_parent = parents[0]

    # Update the best_so_far if the new parent is better
    if best_so_far is None or best_parent[1] > best_so_far[1]:
        best_so_far = best_parent  # Tuple[oracle_output, normalized score, rewards]

    # Append the best score so far for the current generation
    best = max(best_scores_so_far)
    if best_parent[1] > best:
        best_scores_so_far.append(best_parent[1])
        logging.info(f"New best parent found with score: {best_parent[1]}")
    else:
        best_scores_so_far.append(best)

    # Add the performance of the parents to the normalized_scores
    normalized_scores.extend(
        [performance[1] for performance in oracle_offspring_performance]
    )
    rewards.extend([performance[2] for performance in oracle_offspring_performance])

    logging.info(f"Best rewards so far: {best_scores_so_far}")

    return parents, normalized_scores, rewards, best_so_far, best_scores_so_far
