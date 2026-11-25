"""Recurrent action-distribution policy; retains the validated MAPPO actor layout."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from algos.utils.rnn import RNNLayer
from algos.utils.network_init import linear


class ActionDistribution:
    """One joint distribution per agent; the last dimension is the action event."""
    def __init__(self, outputs, layout, log_std=None, masks=None, temperature=1.0):
        if torch.as_tensor(temperature).le(0).any():
            raise ValueError("MAPPO temperature must be positive")
        self.continuous = not layout['category_counts']
        if self.continuous:
            self.normal = Normal(outputs, log_std.clamp(-5, 2).exp() * temperature)
        else:
            logits = outputs / temperature
            if masks is not None:
                logits = logits.masked_fill(~masks.bool(), -torch.inf)
            self.categoricals = []
            for head in logits.split(layout['category_counts'], dim=-1):
                if torch.isneginf(head).all(-1).any():
                    raise ValueError("MAPPO action head has no available category")
                self.categoricals.append(Categorical(logits=head))

    def sample(self, deterministic=False):
        if self.continuous:
            z = self.normal.mean if deterministic else self.normal.sample()
            # Keep stored actions invertible in float32, including saturated samples.
            return z.tanh().clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.stack([d.probs.argmax(-1) if deterministic else d.sample()
                            for d in self.categoricals], dim=-1)

    def log_prob(self, actions):
        if self.continuous:
            a = actions.clamp(-1 + 1e-6, 1 - 1e-6)
            z = torch.atanh(a)
            correction = 2 * (math.log(2) - z - torch.nn.functional.softplus(-2 * z))
            return (self.normal.log_prob(z) - correction).sum(-1, keepdim=True)
        return torch.stack([d.log_prob(actions[..., i].long())
                            for i, d in enumerate(self.categoricals)], -1).sum(-1, keepdim=True)

    def entropy(self):
        if self.continuous:
            # Reparameterized Monte Carlo estimate of transformed entropy.
            z = self.normal.rsample()
            correction = 2 * (math.log(2) - z - torch.nn.functional.softplus(-2 * z))
            return (self.normal.entropy() + correction).sum(-1, keepdim=True)
        return torch.stack([d.entropy() for d in self.categoricals], -1).sum(-1, keepdim=True)


class RecurrentActor(nn.Module):
    def __init__(self, dim_info, use_rnn=True):
        super().__init__()
        self.layout = dim_info.action_layout
        self.hidden_dim = dim_info.hidden_dim
        self.num_layers = dim_info.num_layers
        self.use_rnn = use_rnn
        h = self.hidden_dim
        self.base = nn.Sequential(nn.LayerNorm(dim_info.value), linear(dim_info.value, h),
                                  nn.Tanh(), nn.LayerNorm(h), linear(h, h), nn.Tanh(), nn.LayerNorm(h))
        if use_rnn:
            self.rnn = RNNLayer(h, h, self.num_layers, True)
        output_dim = sum(self.layout['category_counts']) or self.layout['continuous_dim']
        self.head = linear(h, output_dim, 0.01)
        self.log_std = nn.Parameter(torch.zeros(output_dim)) if not self.layout['category_counts'] else None

    def forward(self, inputs, hidden, masks):
        features = self.base(inputs)
        if self.use_rnn:
            features, hidden = self.rnn(features, hidden, masks)
        return self.head(features), hidden

    def distribution(self, outputs, masks=None, temperature=1.0):
        return ActionDistribution(outputs, self.layout, self.log_std, masks, temperature)
