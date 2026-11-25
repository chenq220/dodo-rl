"""Centralized action value networks for MADDPG."""
from torch import nn


def build_critics(n_agents: int, input_dim: int, hidden_dim: int) -> nn.ModuleList:
    return nn.ModuleList(
        nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for _ in range(n_agents)
    )
