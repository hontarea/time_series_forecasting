from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Single-fold trading environment for RL-based execution.

    Each episode corresponds to one ``horizon``-hour forecast horizon. The agent
    receives a (``horizon`` + 5)-dimensional observation at every hour and decides to go
    short (-1), flat (0), or long (+1).

    Observation vector (``horizon`` + 5 dims):
        forecast[0..horizon-1]  - model log-return forecast for each hour
        position                - current position  {-1, 0, 1}
        cum_return              - cumulative log P&L since episode start
        hours_remaining         - (horizon - current_hour) / horizon
        forecast_accuracy       - rolling MAE across recent folds (0 if none)
        volatility              - std of log-returns in the training window

    Action space:  Discrete(3) = {0: -1,  1: 0,  2: +1}

    Args:
        horizon         : steps per episode (default 24)
        txn_cost        : one-way transaction cost (default 0.002)
        carry_position  : whether to carry position across episodes
        reward_fn       : "raw" (step P&L) 
    """

    metadata = {"render_modes": []}

    # Map discrete action to target position
    _ACTION_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(
        self,
        horizon: int = 24,
        txn_cost: float = 0.002,
        carry_position: bool = True,
        reward_fn: str = "raw",
    ) -> None:
        super().__init__()

        self.horizon = horizon
        self.txn_cost = txn_cost
        self.carry_position = carry_position
        assert reward_fn in ("raw"), "reward_fn must be 'raw'"
        self.reward_fn = reward_fn

        # Spaces
        obs_dim = horizon + 5  # 24 forecasts + 5 scalars
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Episode state (initialised in reset)
        self._hour: int = 0
        self._position: float = 0.0
        self._cumulative_return: float = 0.0
        self._forecast: np.ndarray = np.zeros(horizon, dtype=np.float32)
        self._actual_returns: np.ndarray = np.zeros(horizon, dtype=np.float32)
        self._forecast_accuracy: float = 0.0
        self._volatility: float = 0.0


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        options = options or {}

        self._forecast = np.asarray(
            options.get("forecast", np.zeros(self.horizon)), dtype=np.float32
        )

        actual_prices = np.asarray(
            options.get("actual_prices", np.ones(self.horizon + 1)), dtype=np.float64
        )
        self._actual_returns = np.log(
            actual_prices[1:] / actual_prices[:-1]
        ).astype(np.float32)

        if self.carry_position and "carry_position" in options:
            self._position = float(options["carry_position"])
        else:
            self._position = 0.0

        self._hour = 0
        self._cumulative_return = 0.0
        self._forecast_accuracy = float(options.get("forecast_accuracy", 0.0))
        self._volatility = float(options.get("volatility", 0.0))

        return self._obs(), {}


    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        target_pos = self._ACTION_MAP[int(action)]

        cost = self.txn_cost * abs(target_pos - self._position)
        self._position = float(target_pos)

        actual_ret = float(self._actual_returns[self._hour])
        step_pnl = self._position * actual_ret - cost
        self._cumulative_return += step_pnl
        self._hour += 1

        reward = self._compute_reward(step_pnl)
        terminated = self._hour >= self.horizon

        info = {
            "step_pnl": step_pnl,
            "position": self._position,
            "cumulative_return": self._cumulative_return,
        }

        return self._obs(), float(reward), terminated, False, info


    def _compute_reward(self, step_pnl: float) -> float:
        """Compute per-step reward (currently raw step P&L)."""
        if self.reward_fn == "raw":
            return step_pnl

    def _obs(self) -> np.ndarray:
        """Build and return the current observation vector."""
        hours_remaining = (self.horizon - self._hour) / self.horizon
        return np.concatenate([
            self._forecast,
            [self._position, self._cumulative_return, hours_remaining,
             self._forecast_accuracy, self._volatility],
        ]).astype(np.float32)


    def render(self) -> None:
        """Rendering is not implemented; required by Gymnasium interface."""
        pass
