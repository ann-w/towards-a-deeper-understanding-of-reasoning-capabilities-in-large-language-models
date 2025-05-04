import logging
import random
from typing import Tuple

import gym

from scripts.llm_query import get_query

logging.basicConfig(level=logging.INFO)


def initialize_environment(env_name: str, env_steps: int = None, num_iter: int = None):
    env = gym.make(f"smartplay:{env_name}-v0")

    # Set the number of steps to run the environment for
    env_steps = env_steps if env_steps is not None else env.default_steps

    # Set the number of iterations (or episodes) to run the environment for
    num_iter = num_iter if num_iter is not None else env.default_iter

    logging.info(f"env: {env_name}, env_steps: {env_steps}, num_iter: {num_iter}")

    return env, env_steps, num_iter


def initialize_model(model_name: str, model_type: str):
    query_model = get_query(model_name, model_type)
    logging.info(f"model: {model_name}, model_type: {model_type}")
    return query_model


def parse_action_number(response: str, environment: str) -> int:
    try:
        response = response.lower().replace("*", "")

        # Step 1: Look for "Action: ..."
        if "action:" in response:
            # Extract the complete text after "Action:"
            after_action = response.split("action:", 1)[1].strip()

            # Check for numeric action first
            first_word = (
                after_action.split(None, 1)[0].rstrip(".,;:!?") if after_action else ""
            )
            if first_word.isdigit():
                action_number = int(first_word)
                return action_number - 1  # 0-based

            # If not numeric, use the full action text for matching
            action_text = after_action.split("\n")[0].strip()  # Get only the first line

            # Step 2: Match against valid actions
            valid_actions = get_valid_actions_for_env(environment)
            best_match = None
            best_match_idx = -1

            for idx, action_desc in valid_actions.items():
                if action_text.lower() == action_desc.lower():
                    # Exact match
                    return idx - 1
                # Partial match - keep track of the longest one
                elif (
                    action_text.lower() in action_desc.lower()
                    or action_desc.lower() in action_text.lower()
                ):
                    if best_match is None or len(action_desc) > len(best_match):
                        best_match = action_desc
                        best_match_idx = idx

            if best_match_idx != -1:
                return best_match_idx - 1

        # Step 3: Handle responses with <think> sections and Answer/boxed format
        if "<think>" in response and "</think>" in response:
            # Remove the thinking section
            cleaned_response = ""
            in_think_section = False

            for line in response.split("\n"):
                if "<think>" in line:
                    in_think_section = True
                    continue
                if "</think>" in line:
                    in_think_section = False
                    continue
                if not in_think_section:
                    cleaned_response += line + "\n"

            # Look for "Answer:" pattern
            answer_text = None
            if "answer:" in cleaned_response.lower():
                answer_text = cleaned_response.lower().split("answer:", 1)[1].strip()
                answer_text = answer_text.split("\n\n")[0].strip()

            # Look for \boxed{} pattern
            elif "\\boxed{" in cleaned_response:
                boxed_parts = cleaned_response.split("\\boxed{", 1)[1].split("}", 1)
                if len(boxed_parts) > 0:
                    boxed_content = boxed_parts[0].strip()
                    # Check if the boxed content is a number
                    if boxed_content.isdigit():
                        action_number = int(boxed_content)
                        valid_actions = get_valid_actions_for_env(environment)

                        # If the action number is valid for this environment, use it
                        if action_number in valid_actions:
                            return action_number - 1

                        # If not valid, try to find a matching action in surrounding context
                        nearby_text = cleaned_response.lower()
                        for idx, action_desc in valid_actions.items():
                            # Look for action description in the full response
                            if action_desc.lower() in nearby_text:
                                return idx - 1

                        # If no contextual match found, return the boxed number
                        # (it might be valid in 0-indexed form)
                        return action_number - 1
                    else:
                        answer_text = boxed_content

            if answer_text:
                valid_actions = get_valid_actions_for_env(environment)
                best_match = None
                best_match_idx = -1

                for idx, action_desc in valid_actions.items():
                    if answer_text.lower() == action_desc.lower():
                        # Exact match
                        return idx - 1
                    # Partial match - keep track of the longest one
                    elif (
                        answer_text.lower() in action_desc.lower()
                        or action_desc.lower() in answer_text.lower()
                    ):
                        if best_match is None or len(action_desc) > len(best_match):
                            best_match = action_desc
                            best_match_idx = idx

                if best_match_idx != -1:
                    return best_match_idx - 1

        # Step 4: No match => invalid
        return -1
    except (IndexError, ValueError):
        return -1


def convert_parsed_action_to_valid_index(
    parsed_action_number: int, env: gym.Env
) -> Tuple[str, bool]:

    invalid_action = False

    # Check if the action index is within the valid range
    if parsed_action_number < 0 or parsed_action_number >= len(env.action_list):
        invalid_action = True
        action_index = random.randint(0, len(env.action_list) - 1)
    else:
        action_index = parsed_action_number

    return action_index, invalid_action


def get_valid_actions_for_env(environment: str) -> dict:
    """
    Retrieve the valid actions dictionary based on the correct environment name.
    """
    # Common action sets
    hanoi_actions = {
        1: "Move the top disk of rod A to the top of rod B",
        2: "Move the top disk of rod A to the top of rod C",
        3: "Move the top disk of rod B to the top of rod A",
        4: "Move the top disk of rod B to the top of rod C",
        5: "Move the top disk of rod C to the top of rod A",
        6: "Move the top disk of rod C to the top of rod B",
    }

    messenger_actions = {
        1: "Move North",
        2: "Move South",
        3: "Move West",
        4: "Move East",
        5: "Do Nothing",
    }

    action_dicts = {
        "RockPaperScissorBasic": {1: "Rock", 2: "Paper", 3: "Scissor"},
        "BanditTwoArmedHighLowFixed": {
            1: "Pull slot machine 1",
            2: "Pull slot machine 2",
        },
        "Crafter": {
            1: "Move West: Flat ground west of the agent.",
            2: "Move East: Flat ground east of the agent.",
            3: "Move North: Flat ground north of the agent.",
            4: "Move South: Flat ground south of the agent.",
            5: "Do: Facing creature or material; have necessary tool.",
            6: "Sleep: Energy level is below maximum.",
            7: "Place Stone: Stone in inventory.",
            8: "Place Table: Wood in inventory.",
            9: "Place Furnace: Stone in inventory.",
            10: "Place Plant: Sapling in inventory.",
            11: "Make Wood Pickaxe: Nearby table; wood in inventory.",
            12: "Make Stone Pickaxe: Nearby table; wood, stone in inventory.",
            13: "Make Iron Pickaxe: Nearby table, furnace; wood, coal, iron in inventory.",
            14: "Make Wood Sword: Nearby table; wood in inventory.",
            15: "Make Stone Sword: Nearby table; wood, stone in inventory.",
            16: "Make Iron Sword: Nearby table, furnace; wood, coal, iron in inventory.",
            17: "Noop: Always applicable.",
        },
    }

    # Add all Hanoi variants
    hanoi_variants = [
        "Hanoi2Disk",
        "Hanoi3Disk",
        "Hanoi2DShowValid",
        "Hanoi3DShowValid",
        "Hanoi2DRewardShaping",
        "Hanoi3DRewardShaping",
        "Hanoi2DShowValidRewardShaping",
        "Hanoi3DShowValidRewardShaping",
    ]
    for variant in hanoi_variants:
        action_dicts[variant] = hanoi_actions

    # Add all Messenger variants
    messenger_variants = [
        "MessengerL1",
        "MessengerL2",
        "MessengerL1Shaped",
        "MessengerL1NoRand",
        "MessengerL1ShapedNoRand",
    ]
    for variant in messenger_variants:
        action_dicts[variant] = messenger_actions

    return action_dicts.get(environment, {})
