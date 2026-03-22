from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tsf.agents.base import BaseAgent
from tsf.agents.replay_buffer import ReplayBuffer


class _QNetwork(nn.Module):
    """MLP Q-function: state_dim -> hidden_dims -> n_actions."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with experience replay and a target network.

    Args:
        state_dim: dimensionality of the observation vector
        n_actions: number of discrete actions (default 3)
        hidden_dims: hidden layer sizes (default [128, 64])
        lr: Adam learning rate (default 1e-4)
        gamma: discount factor (default 0.99)
        epsilon_start: initial exploration rate (default 1.0)
        epsilon_end: minimum exploration rate (default 0.05)
        epsilon_decay_steps: linear decay schedule length (default 5000)
        buffer_size: replay buffer capacity (default 10000)
        batch_size: mini-batch size (default 32)
        target_update_freq: update target net every N learn() calls
        device: torch device string (default "cuda" if available)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        hidden_dims: List[int] = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
        buffer_size: int = 10_000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self._online = _QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self._target = _QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._optimizer = optim.Adam(self._online.parameters(), lr=lr)
        self._buffer = ReplayBuffer(buffer_size)
        self._learn_steps: int = 0

    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        """Select an action with epsilon-greedy exploration or greedy exploitation."""
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self._online(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition to the replay buffer."""
        self._buffer.push(state, action, reward, next_state, done)

    def learn(self) -> dict:
        """Sample a batch and perform one gradient update; returns loss and epsilon metrics."""
        if len(self._buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self._buffer.sample(self.batch_size)

        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        s_next = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q values for taken actions
        q_online = self._online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q values (no gradient)
        with torch.no_grad():
            q_target_next = self._target(s_next).max(dim=1).values
            y = r + self.gamma * q_target_next * (1.0 - d)

        loss = nn.functional.mse_loss(q_online, y)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._learn_steps += 1

        # Sync target network
        if self._learn_steps % self.target_update_freq == 0:
            self._target.load_state_dict(self._online.state_dict())

        # Decay epsilon linearly
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        return {
            "loss": float(loss.item()),
            "epsilon": self.epsilon,
            "buffer_size": len(self._buffer),
        }

    def save(self, path: Path) -> None:
        """Save online/target network weights, optimizer state, and training counters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self._online.state_dict(),
                "target_state_dict": self._target.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self._learn_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Restore network weights, optimizer state, and training counters from a checkpoint."""
        ckpt = torch.load(Path(path), map_location=self.device)
        self._online.load_state_dict(ckpt["online_state_dict"])
        self._target.load_state_dict(ckpt["target_state_dict"])
        self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self._learn_steps = ckpt["learn_steps"]
