"""Monotonic, state-conditioned mixing for QMIX."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class QMixer(nn.Module):
    """Mix individual utilities using nonnegative state-conditioned weights."""

    def __init__(self, n_agents: int, state_shape, embed_dim: int, hypernet_layers: int, hypernet_embed: int):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = math.prod(state_shape) if isinstance(state_shape, (tuple, list)) else int(state_shape)
        self.embed_dim = embed_dim
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, embed_dim * n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, embed_dim * n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, embed_dim),
            )
        else:
            raise ValueError(f"QMIX hypernet_layers must be 1 or 2, got {hypernet_layers}")
        self.hyper_b_1 = nn.Linear(self.state_dim, embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, utilities, states):
        batch_size = utilities.size(0)
        states = states.reshape(-1, self.state_dim)
        utilities = utilities.reshape(-1, 1, self.n_agents)
        first_weights = self.hyper_w_1(states).abs().reshape(-1, self.n_agents, self.embed_dim)
        first_bias = self.hyper_b_1(states).reshape(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(utilities, first_weights) + first_bias)
        final_weights = self.hyper_w_final(states).abs().reshape(-1, self.embed_dim, 1)
        value = self.V(states).reshape(-1, 1, 1)
        return (torch.bmm(hidden, final_weights) + value).reshape(batch_size, -1, 1)
