"""Recurrent action-value and policy execution with fixed agent slots."""
from __future__ import annotations

import torch

from .recurrent_q import IndependentUtilities, RecurrentUtility


class RecurrentController:
    shared = True

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type
        input_dim = scheme["obs"]["vshape"]
        if args.obs_last_action:
            input_dim += scheme["actions_onehot"]["vshape"][0]
        if args.obs_agent_id:
            input_dim += self.n_agents
        use_rnn = args.algo_config.use_rnn
        if self.shared:
            self.agent = RecurrentUtility(input_dim, args.hidden_dim, args.n_actions, use_rnn)
        else:
            self.agent = IndependentUtilities(input_dim, args.hidden_dim, args.n_actions, self.n_agents, use_rnn)
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def forward(self, batch, t, test_mode=False):
        values = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t > 0:
                previous = batch["actions_onehot"][:, t - 1]
            elif self.shared:
                try:
                    previous = batch["initial_actions_onehot"]
                except (KeyError, ValueError):
                    previous = torch.zeros_like(batch["actions_onehot"][:, t])
            else:
                previous = torch.zeros_like(batch["actions_onehot"][:, t])
            values.append(previous)
        if self.args.obs_agent_id:
            identity = torch.eye(self.n_agents, device=batch.device)
            values.append(identity.unsqueeze(0).expand(batch.batch_size, -1, -1))
        inputs = torch.cat([value.reshape(batch.batch_size * self.n_agents, -1) for value in values], dim=1)
        output, self.hidden_states = self.agent(inputs, self.hidden_states)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args.algo_config, "mask_before_softmax", True):
                available = batch["avail_actions"][:, t].reshape(batch.batch_size * self.n_agents, -1)
                output[available == 0] = -1e10
            output = torch.softmax(output, dim=-1)
        return output.reshape(batch.batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other):
        self.agent.load_state_dict(other.agent.state_dict())


class IndependentRecurrentController(RecurrentController):
    shared = False
