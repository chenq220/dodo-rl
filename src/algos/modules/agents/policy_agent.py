from __future__ import annotations

import numpy as np
import torch
from algos.agents import Agent, record_action_warp
from algos.modules.agents.policy import RecurrentActor
from algos.utils.action_masks import action_mask_vector
from common.plugins import agent_plugin
from algos.utils.recurrent_inputs import RecurrentAgentInputs
from config import PolicyRnnAgentConfig
from utils.checkpoint import validated_checkpoint_state_dict


@agent_plugin(PolicyRnnAgentConfig, default_obs_agent_id=True)
class PolicyRNNAgent(RecurrentAgentInputs, Agent):
    """输出动作分布的循环策略 Agent，固定使用策略采样语义。"""
    fixed_output_type = "pi_logits"
    supported_action_spaces = ('discrete', 'multi_discrete', 'continuous', 'multi_continuous')

    def build_empty_network(self, dim_info, **kwargs):
        return RecurrentActor(dim_info, self.config.algo_config.use_rnn)

    def build_network_from_checkpoint(self, checkpoint, dim_info, **kwargs):
        net = self.build_empty_network(dim_info)
        weights = validated_checkpoint_state_dict(net, checkpoint, agent_name=self.name,
                                                 model_name='agent_net', expected_dimensions={})
        return net, weights

    @record_action_warp
    def get_actions(self, obs, agent_list, hidden_in, action_mask=None, epsilon=0.0,
                    evaluate=False, last_actions=None, temperature=1.0, deterministic=False, **kwargs):
        self.agent_net.eval()
        inputs = self.inputs(obs, last_actions, agent_list)
        hidden = torch.stack([hidden_in[a].reshape(self.config.num_layers, self.config.hidden_dim)
                              for a in agent_list]).to(self.device)
        masks = None
        if self.action_layout['category_counts']:
            masks = torch.as_tensor(np.stack([action_mask_vector(
                self.action_space, None if action_mask is None else action_mask.get(a))
                for a in agent_list]), device=self.device)
        with torch.no_grad():
            outputs, hidden_out = self.agent_net(inputs, hidden, torch.ones(len(agent_list), 1, device=self.device))
            dist = self.agent_net.distribution(outputs, masks, temperature)
            values = dist.sample(evaluate or deterministic)
            logs = dist.log_prob(values).flatten()
        self._on_policy_stats = ({**{a: float(logs[i]) for i, a in enumerate(agent_list)},
                                  '__temperature__': float(temperature)}, None)
        self._policy_temperature = float(temperature)
        values = values.cpu().numpy()
        actions = {a: self.action_space.decode(int(values[i, 0]) if self.action_space.kind == 'discrete'
                                              else values[i]) for i, a in enumerate(agent_list)}
        return actions, {a: hidden_out[i].detach() for i, a in enumerate(agent_list)}

    def consume_on_policy_stats(self):
        stats = getattr(self, "_on_policy_stats", (None, None))
        self._on_policy_stats = (None, None)
        return stats
