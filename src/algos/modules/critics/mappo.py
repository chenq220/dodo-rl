"""Centralized state value network for MAPPO."""
from torch import nn

from algos.utils.network_init import linear


class CentralValue(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), linear(input_dim, hidden_dim), nn.Tanh(),
                                 nn.LayerNorm(hidden_dim), linear(hidden_dim, hidden_dim), nn.Tanh(),
                                 nn.LayerNorm(hidden_dim), linear(hidden_dim, 1, 1.0))

    def forward(self, inputs):
        return self.net(inputs)
