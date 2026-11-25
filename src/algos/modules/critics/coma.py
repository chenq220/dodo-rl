"""Centralized action value network for COMA."""
import torch
from torch import nn


class JointActionCritic(nn.Module):
    """Centralized team action value with an explicit active-slot mask."""

    def __init__(self, state_dim, n_agents, action_dim, hidden_dim):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_agents * (action_dim + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, actions, active):
        joint_actions = actions * active.unsqueeze(-1)
        inputs = torch.cat((state, joint_actions.flatten(-2), active.float()), dim=-1)
        return self.net(inputs).squeeze(-1)
