"""Decentralized execution for bounded continuous MADDPG policies."""
from __future__ import annotations

import math

import torch
from torch import nn

from algos.agents import Agent, record_action_warp
from common.plugins import agent_plugin
from config import AgentOutputType, MaddpgAgentConfig
from utils.checkpoint import extract_state_dict, validated_checkpoint_state_dict
from algos.utils.recurrent_inputs import RecurrentAgentInputs


_MIN_LOG_STD = -5.0
_MAX_LOG_STD = 2.0


class _GaussianActionHead(nn.Linear):
    """Keep the mean weights in the original linear layer's checkpoint keys."""

    def __init__(self, input_dim, action_dim, initial_log_std):
        super().__init__(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, initial_log_std)

    def forward(self, features):
        return super().forward(features), self.log_std(features)


class MaddpgActor(nn.Module):
    """One deterministic policy with Gaussian rollout exploration per agent slot."""

    def __init__(self, dimensions, initial_std):
        super().__init__()
        self.initial_log_std = math.log(max(initial_std, math.exp(_MIN_LOG_STD)))
        self.policies = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dimensions.value, dimensions.hidden_dim),
                nn.ReLU(),
                nn.Linear(dimensions.hidden_dim, dimensions.hidden_dim),
                nn.ReLU(),
                _GaussianActionHead(
                    dimensions.hidden_dim, dimensions.action_layout["value_dim"], self.initial_log_std,
                ),
            )
            for _ in range(dimensions.n_agents)
        )

    def forward(self, inputs, slots=None):
        if slots is None:
            slots = range(len(self.policies))
        means, log_stds = zip(
            *(self.policies[slot](inputs[..., index, :]) for index, slot in enumerate(slots)),
        )
        return torch.stack(means, dim=-2), torch.stack(log_stds, dim=-2)

    @staticmethod
    def sample_actions(mean, log_std, *, exploration_std=None):
        """Sample exploratory rollout actions, optionally with config-owned noise."""
        if exploration_std is None:
            # Compatibility for callers that still own a learned Gaussian head.
            std = log_std.clamp(_MIN_LOG_STD, _MAX_LOG_STD).exp()
        else:
            std = torch.as_tensor(exploration_std, dtype=mean.dtype, device=mean.device)
        return (mean + std * torch.randn_like(mean)).tanh()

    @staticmethod
    def deterministic_actions(mean):
        """Return the bounded policy action used by DPG updates and evaluation."""
        return mean.tanh()


def validated_actor_state_dict(actor, checkpoint, *, agent_name):
    """Initialize the scale head when loading a deterministic MADDPG checkpoint."""
    try:
        given = extract_state_dict(checkpoint)
    except ValueError:
        given = {}
    expected = actor.state_dict()
    scale_keys = {key for key in expected if ".log_std." in key}
    if given.keys() == expected.keys() - scale_keys:
        checkpoint = {
            **given,
            **{
                key: torch.full_like(expected[key], actor.initial_log_std if key.endswith(".bias") else 0.0)
                for key in scale_keys
            },
        }
    return validated_checkpoint_state_dict(
        actor, checkpoint, agent_name=agent_name, model_name="agent_net",
        expected_dimensions={"n_agents": len(actor.policies)},
    )


@agent_plugin(MaddpgAgentConfig, default_obs_agent_id=False)
class MaddpgAgent(RecurrentAgentInputs, Agent):
    """Executes each slot's policy using only its own inputs."""

    fixed_output_type = AgentOutputType.CONTINUOUS_ACTION.value
    supported_action_spaces = ("continuous", "multi_continuous")

    def build_empty_network(self, dim_info, **kwargs):
        algo_config = self.config.algo_config
        std = algo_config.exploration_std
        if not math.isfinite(std) or std < 0:
            raise ValueError("MADDPG exploration_std must be finite and nonnegative")
        final_std = algo_config.exploration_std_final
        if final_std is not None and (not math.isfinite(final_std) or final_std < 0):
            raise ValueError("MADDPG exploration_std_final must be finite and nonnegative or None")
        return MaddpgActor(dim_info, std)

    def build_network_from_checkpoint(self, checkpoint, dim_info, **kwargs):
        network = self.build_empty_network(dim_info)
        weights = validated_actor_state_dict(network, checkpoint, agent_name=self.name)
        return network, weights

    @record_action_warp
    @torch.no_grad()
    def get_actions(self, obs, agent_list, hidden_in, action_mask=None, epsilon=0.0,
                    evaluate=False, last_actions=None, deterministic=False, *, temperature=1.0, **kwargs):
        if not agent_list:
            return {}, {}
        inputs = self.inputs(obs, last_actions, agent_list)
        slots = [self.agent_list.index(agent_id) for agent_id in agent_list]
        mean, log_std = self.agent_net(inputs, slots)
        config = self.config.algo_config
        exploration_std = config.exploration_std
        if config.exploration_std_final is not None:
            fraction = min(1.0, max(0.0, float(epsilon)))
            exploration_std = (
                config.exploration_std_final
                + fraction * (config.exploration_std - config.exploration_std_final)
            )
        values = (self.agent_net.deterministic_actions(mean)
                  if evaluate or deterministic else self.agent_net.sample_actions(
                      mean, log_std, exploration_std=exploration_std,
                  ))
        return dict(zip(agent_list, map(self.action_space.decode, values.cpu().numpy()))), {}
