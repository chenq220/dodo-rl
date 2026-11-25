"""Shared initialization for actor and value networks."""
import math

from torch import nn


def linear(in_dim, out_dim, gain=math.sqrt(2)):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer
