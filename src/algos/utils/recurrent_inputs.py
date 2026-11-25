"""Input encoding and per-agent recurrent state shared by policy and Q agents."""
import numpy as np
import torch


class RecurrentAgentInputs:
    def get_init_hidden_state(self):
        return {a: torch.zeros(self.config.num_layers, self.config.hidden_dim, device=self.device)
                for a in self.agent_list}

    def inputs(self, obs, last_actions, agent_ids):
        rows = []
        for a in agent_ids:
            parts = [np.asarray(obs[a], dtype=np.float32).flatten()]
            if self.config.obs_last_action:
                previous = None if last_actions is None else last_actions.get(a)
                parts.append(self.action_space.encode_previous_action(previous))
            if self.config.obs_agent_id:
                parts.append(np.eye(self.n_agents, dtype=np.float32)[self.agent_list.index(a)])
            rows.append(np.concatenate(parts))
        return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=self.device)
