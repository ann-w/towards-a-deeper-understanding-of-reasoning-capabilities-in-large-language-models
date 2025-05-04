import logging
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
from smartplay.eval import normalize_score

from scripts.compose_ingame_prompt import (generate_agent_prompt,
                                           get_planner_output,
                                           get_reflection_output)
from scripts.constants import ENV_TRAJECTORY_SAMPLES
from scripts.memory import Memory
from scripts.utils.environment_utils import (
    convert_parsed_action_to_valid_index, parse_action_number)
from scripts.utils.timeit import timeit
from scripts.utils.wandb_logging import (create_episode_table,
                                         finalize_episode_logging,
                                         log_episode_metrics)


@timeit
def run_single_episode(
    env,
    env_name: str,
    env_steps: int,
    query_model: Callable,
    model_type: str,
    memory: Memory,
    global_step: int,
    reflection_interval_steps: int,
    episode_number: Optional[int] = None,
    oracle_output: str = None,
    use_reflection: bool = False,
    reflection_start_step: int = 10,
    count_token_usage: bool = False,
    use_planner: bool = False,
) -> Tuple[float, List[float]]:

    rewards = []
    _, info = env.reset()
    reflection_output = None
    planner_output = None
    planner_prompt = None
    action = None

    # Get the last n transitions from the memory
    if env_name not in ENV_TRAJECTORY_SAMPLES:
        raise ValueError(f"Environment {env_name} not found in env_history_samples")

    # Get the number of trajectory samples from the dictionary
    n_trajectory_samples = ENV_TRAJECTORY_SAMPLES[env_name]

    episode_table = create_episode_table()

    logging.info("n_trajectory_samples for history: %s", n_trajectory_samples)

    for step_number in range(env_steps):
        agent_prompt = ""
        answer = ""
        num_input_tokens = 0
        num_output_tokens = 0
        action = None
        invalid_action = False

        state = info["obs"]
        trajectory = memory.get_transitions(n=n_trajectory_samples)

        # Generate the reflection output if reflection is enabled
        if use_reflection and step_number + global_step >= reflection_start_step:
            if step_number % reflection_interval_steps == 0:
                reflection_output = get_reflection_output(
                    info=info,
                    trajectory=trajectory,
                    model_type=model_type,
                    query_model=query_model,
                )

        # Ask for predictions from the planner
        if use_planner:
            planner_prompt, planner_output = get_planner_output(
                info=info,
                trajectory=trajectory,
                model_type=model_type,
                query_model=query_model,
                reflection_output=reflection_output,
            )

            # Parse planner action directly without generating agent prompt
            action_index = parse_action_number(planner_output, env_name)

            # Mark as invalid if parse failed OR if index is out of range
            invalid_action = (
                action_index == -1
                or action_index >= len(env.action_list)
                or action_index < 0
            )

            if invalid_action:
                # Get random action and log the invalid action issue
                action_index = random.randint(0, len(env.action_list) - 1)
                logging.warning(
                    f"Invalid planner action index {action_index}, using random action instead"
                )

        else:
            # Generate the appropriate prompt based on reflection and Oracle usage
            agent_prompt = generate_agent_prompt(
                info=info,
                trajectory=trajectory,
                model_type=model_type,
                use_reflection_template=use_reflection,
                step_number=step_number + global_step,
                reflection_start_step=reflection_start_step,
                reflection_output=reflection_output,
                use_oracle_template=bool(
                    oracle_output
                ),  # Check if oracle_output is provided
                oracle_output=oracle_output,
                planner_output=planner_output,
            )

            answer, num_input_tokens, num_output_tokens = query_model(
                agent_prompt, count_token_usage
            )
            parsed_action_number = parse_action_number(answer, environment=env_name)
            action_index, invalid_action = convert_parsed_action_to_valid_index(
                parsed_action_number, env
            )
            action = env.action_list[action_index]

        _, reward, done, info = env.step(action_index)
        score = info["score"]

        next_state = info["obs"]
        done_bool = True if done == 1 else False

        memory.store_transition((state, action, reward, next_state, done_bool))
        memory.store_reflection((reflection_output))
        rewards.append(reward)

        log_episode_metrics(
            step=step_number,
            prompt=agent_prompt,
            observation=state,
            score=score,
            reward=reward,
            total_reward=sum(rewards),
            answer=answer,
            action_index=action_index,
            action=action,
            invalid_action=invalid_action,
            reflection_output=reflection_output if reflection_output else "",
            oracle_output=(
                oracle_output[0] if isinstance(oracle_output, tuple) else oracle_output
            ),
            planner_prompt=planner_prompt if planner_prompt else "",
            planner_output=planner_output if planner_output else "",
            wandb_table=episode_table,
            input_tokens=num_input_tokens if num_input_tokens else "",
            output_tokens=num_output_tokens if num_output_tokens else "",
            model_type=model_type,
        )

        if done:
            break

    # Normalizes the score between 0 and 1 based on predefined minum and human scores for the game
    normalized_score = normalize_score(env_name, score)

    finalize_episode_logging(episode_number, episode_table, rewards)

    logging.info(f"Episode statistics")
    logging.info(
        f"Total reward: {np.sum(rewards)}, Normalized score: {round(normalized_score, 2)}, Score: {score}"
    )

    return normalized_score, np.sum(rewards), score
