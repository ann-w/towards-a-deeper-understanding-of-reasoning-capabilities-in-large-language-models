import random
from collections import deque
from typing import Any, Dict, List, Tuple, Union


class Memory:
    """
    A class for managing the storage of transitions and reflections for an intelligent agent.

    Attributes:
        transition_memory: A deque for storing transitions.
        reflection_memory: A deque for storing reflections.
    """

    def __init__(self, capacity: int = None):
        self.transition_memory = deque(maxlen=capacity) if capacity else deque()
        self.reflection_memory = deque(maxlen=capacity) if capacity else deque()

    def store_transition(
        self, transition: Tuple[Union[str, float], int, Union[str, float], bool]
    ) -> None:
        """Store a transition in the transition memory.

        Each transition is a tuple of the form (state, action_index, reward, next_state, done).
        """
        self.transition_memory.append(transition)

    def store_reflection(self, reflection: Dict[str, Any]) -> None:
        """Store a reflection in the reflection memory.

        Each reflection is a dictionary containing 'context', 'recommended_action', and 'reasoning'.
        """
        self.reflection_memory.append(reflection)

    def get_transitions(
        self, n: int = None
    ) -> List[Tuple[Union[str, float], int, Union[str, float], bool]]:
        """Return the last n transitions from the transition memory. If n is None, return all."""
        if n is None:
            return list(self.transition_memory)
        return list(self.transition_memory)[-n:]

    def get_reflections(self, n: int = None) -> List[Dict[str, Any]]:
        """Return the last n reflections from the reflection memory. If n is None, return all."""
        if n is None:
            return list(self.reflection_memory)
        return list(self.reflection_memory)[-n:]

    def sample_transitions(
        self, batch_size: int
    ) -> List[Tuple[Tuple[Union[str, float], int, Union[str, float], bool]]]:
        """Sample a batch of transitions from the transition memory."""
        if batch_size > len(self.transition_memory):
            raise ValueError(
                "Batch size is larger than the number of stored transitions."
            )
        return random.sample(list(self.transition_memory), batch_size)

    def sample_reflections(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of reflections from the reflection memory."""
        if batch_size > len(self.reflection_memory):
            raise ValueError(
                "Batch size is larger than the number of stored reflections."
            )
        return random.sample(list(self.reflection_memory), batch_size)

    def clear_transitions(self) -> None:
        """Clear all transitions from the transition memory."""
        self.transition_memory.clear()

    def clear_reflections(self) -> None:
        """Clear all reflections from the reflection memory."""
        self.reflection_memory.clear()

    def __len__(self) -> int:
        """Return the total number of items in both memories."""
        return len(self.transition_memory) + len(self.reflection_memory)
