from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAgent(ABC):
    """Abstract interface for all RL trading agents."""

    @abstractmethod
    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        """Return an action integer given an observation."""

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a (s, a, r, s', done) transition in the replay buffer."""        

    @abstractmethod
    def learn(self) -> dict:
        """Sample a batch and update the agent. Returns a metrics dict."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist agent state to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore agent state from disk."""