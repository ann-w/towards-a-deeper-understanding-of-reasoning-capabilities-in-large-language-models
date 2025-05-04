import concurrent.futures
import logging
from typing import Callable, List, Tuple

import numpy as np

from scripts.episode_runner import run_single_episode
from scripts.memory import Memory


def evaluate_single_oracle_output(
    env,
    env_name: str,
    env_steps: int,
    query_model: Callable,
    model_type: str,
    num_iter: int,
    global_step: int,
    oracle_output: str,
    count_token_usage: bool,
    memory_capacity: int,
    use_reflection: bool,
    reflection_start_step: int,
    reflection_interval_steps: int,
    episode_num: int,
    use_planner: bool,
) -> Tuple[str, float, float]:

    memory = Memory(capacity=memory_capacity)
    normalized_scores = []

    for episode in range(num_iter):

        current_episode_number = episode_num + episode

        normalized_score, rewards, score = run_single_episode(
            env=env,
            env_name=env_name,
            env_steps=env_steps,
            model_type=model_type,
            query_model=query_model,
            global_step=global_step,
            episode_number=current_episode_number,
            memory=memory,
            oracle_output=oracle_output,
            use_reflection=use_reflection,
            reflection_start_step=reflection_start_step,
            reflection_interval_steps=reflection_interval_steps,
            count_token_usage=count_token_usage,
            use_planner=use_planner,
        )
        normalized_scores.append(normalized_score)

    return oracle_output, np.average(normalized_scores), rewards, score


def evaluate_oracle_offspring(
    env,
    env_name: str,
    env_steps: int,
    query_model: Callable,
    model_type: str,
    num_iter: int,
    global_step: int,
    oracle_offspring: List[str],
    memory_capacity: int,
    use_reflection: bool,
    reflection_start_step: int,
    reflection_interval_steps: int,
    max_workers: int,
    episode_num: int,  # global ep number
    count_token_usage: bool,
    use_planner: bool,
) -> List[Tuple[str, float, float]]:
    logging.info("Evaluating Oracle offspring.")

    def run_episode_for_oracle_output(
        oracle_output: str, offspring_id: int, count_token_usage
    ) -> Tuple[str, float]:

        current_episode_number = episode_num + offspring_id * num_iter

        oracle_output, avg_normalized_score, rewards, score = (
            evaluate_single_oracle_output(
                env=env,
                query_model=query_model,
                env_name=env_name,
                env_steps=env_steps,
                num_iter=num_iter,
                global_step=global_step,
                oracle_output=oracle_output,
                model_type=model_type,
                memory_capacity=memory_capacity,
                use_reflection=use_reflection,
                reflection_start_step=reflection_start_step,
                reflection_interval_steps=reflection_interval_steps,
                episode_num=current_episode_number,
                count_token_usage=count_token_usage,
                use_planner=use_planner,
            )
        )
        return oracle_output, avg_normalized_score, rewards, score

    # Parallel evaluation

    results = []

    if max_workers == 1:
        # Sequential processing
        for idx, oracle_output in enumerate(oracle_offspring):
            try:
                oracle_output, average_normalized_score, rewards, score = (
                    run_episode_for_oracle_output(oracle_output, idx, count_token_usage)
                )
                results.append((oracle_output, average_normalized_score, rewards))
            except Exception as e:
                logging.error(
                    f"Error processing output {oracle_output}: {e}", exc_info=True
                )
    else:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each oracle_output with its unique ID to the executor for parallel evaluation
            futures = {
                executor.submit(
                    run_episode_for_oracle_output, oracle_output, idx, count_token_usage
                ): oracle_output
                for idx, oracle_output in enumerate(oracle_offspring)
            }

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                output = futures[
                    future
                ]  # Cache the oracle_output corresponding to this future
                try:
                    # Retrieve the result
                    oracle_output, average_normalized_score, rewards = future.result()
                    results.append((oracle_output, average_normalized_score, rewards))
                except Exception as e:
                    logging.error(
                        f"Error processing future for output {output}: {e}",
                        exc_info=True,
                    )

    return results
