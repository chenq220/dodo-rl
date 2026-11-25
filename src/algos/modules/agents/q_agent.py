"""Action-value execution Agent; learner integration is intentionally separate."""
from __future__ import annotations

import numpy as np
import torch
from algos.agents import Agent, record_action_warp
from algos.modules.agents.recurrent_q import RecurrentQNetwork
from algos.utils.action_masks import action_mask_vector
from common.plugins import agent_plugin
from config import QAgentConfig
from utils.checkpoint import validated_checkpoint_state_dict
from algos.utils.recurrent_inputs import RecurrentAgentInputs


@agent_plugin(QAgentConfig, default_obs_agent_id=True)
class QAgent(RecurrentAgentInputs, Agent):
    """固定输出离散动作 Q 值，以合法动作上的 epsilon-greedy 执行。"""
    fixed_output_type = 'q'
    supported_action_spaces = ('discrete',)

    def build_empty_network(self, dim_info, **kwargs):
        return RecurrentQNetwork(dim_info, self.config.algo_config.use_rnn)

    def build_network_from_checkpoint(self, checkpoint, dim_info, **kwargs):
        net = self.build_empty_network(dim_info)
        weights = validated_checkpoint_state_dict(
            net, checkpoint, agent_name=self.name, model_name='agent_net',
            expected_dimensions={},
        )
        return net, weights

    def get_q_values(self, obs, agent_list, hidden_in, last_actions=None):
        """Return unnormalized action values and independent recurrent state per agent."""
        self.agent_net.eval()
        inputs = self.inputs(obs, last_actions, agent_list)
        hidden = torch.stack([
            torch.as_tensor(hidden_in[a], dtype=torch.float32, device=self.device).reshape(
                self.config.num_layers, self.config.hidden_dim,
            ) for a in agent_list
        ])
        with torch.no_grad():
            values, hidden_out = self.agent_net(inputs, hidden)
        return ({a: values[i].detach() for i, a in enumerate(agent_list)},
                {a: hidden_out[i].detach() for i, a in enumerate(agent_list)})

    @record_action_warp
    def get_actions(self, obs, agent_list, hidden_in, action_mask=None, epsilon=0.0,
                    evaluate=False, last_actions=None, temperature=1.0,
                    deterministic=False, **kwargs):
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError('QAgent epsilon must lie in [0, 1]')
        q_values, hidden_out = self.get_q_values(obs, agent_list, hidden_in, last_actions)
        actions = {}
        for a in agent_list:
            mask = action_mask_vector(self.action_space, None if action_mask is None else action_mask.get(a))
            available = np.flatnonzero(mask)
            if not len(available):
                raise ValueError(f'QAgent has no available action for {a}')
            values = q_values[a].cpu().numpy()
            if not np.isfinite(values[available]).all():
                raise FloatingPointError(f'QAgent has nonfinite action values for {a}')
            if not (evaluate or deterministic) and np.random.random() < epsilon:
                index = int(np.random.choice(available))
            else:
                index = int(available[np.argmax(values[available])])
            actions[a] = self.action_space.decode(index)
        return actions, hidden_out
