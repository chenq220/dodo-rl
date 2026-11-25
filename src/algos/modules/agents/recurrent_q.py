"""Recurrent action-value networks used by the discrete agents."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentUtility(nn.Module):
    """A single utility network with the existing RNN checkpoint layout."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, use_rnn: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_rnn = use_rnn
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim) if use_rnn else nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def init_hidden(self):
        return self.fc1.weight.new_zeros((1, self.hidden_dim))

    def forward(self, inputs, hidden):
        features = F.relu(self.fc1(inputs))
        if self.use_rnn:
            hidden = self.rnn(features, hidden.reshape(-1, self.hidden_dim))
        else:
            hidden = F.relu(self.rnn(features))
        return self.fc2(hidden), hidden


class IndependentUtilities(nn.Module):
    """One utility network per fixed agent slot."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, n_agents: int, use_rnn: bool = True):
        super().__init__()
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.agents = nn.ModuleList(
            RecurrentUtility(input_dim, hidden_dim, action_dim, use_rnn)
            for _ in range(n_agents)
        )

    def init_hidden(self):
        return torch.cat([agent.init_hidden() for agent in self.agents])

    def forward(self, inputs, hidden):
        inputs = inputs.reshape(-1, self.n_agents, self.input_dim)
        values, next_hidden = [], []
        for index, agent in enumerate(self.agents):
            output, state = agent(inputs[:, index], hidden[:, index])
            values.append(output.unsqueeze(1))
            next_hidden.append(state.unsqueeze(1))
        return torch.cat(values, dim=1).reshape(-1, values[0].shape[-1]), torch.cat(next_hidden, dim=1)


class RecurrentQNetwork(nn.Module):
    def __init__(self, dim_info, use_rnn=True):
        super().__init__()
        self.hidden_dim = dim_info.hidden_dim
        self.num_layers = dim_info.num_layers
        self.use_rnn = use_rnn
        self.fc1 = nn.Linear(dim_info.value, self.hidden_dim)
        if use_rnn:
            self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers)
        else:
            self.rnn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, dim_info.action_dim)

    def forward(self, inputs, hidden):
        """One decision: inputs [N,D], hidden [N,L,H], Q values [N,A]."""
        features = torch.relu(self.fc1(inputs))
        if self.use_rnn:
            features, next_hidden = self.rnn(
                features.unsqueeze(0), hidden.transpose(0, 1).contiguous(),
            )
            features = features.squeeze(0)
            next_hidden = next_hidden.transpose(0, 1)
        else:
            features = torch.relu(self.rnn(features))
            next_hidden = hidden
        return self.fc2(features), next_hidden
