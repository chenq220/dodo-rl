"""Batch views and recurrent actor evaluation shared by policy learners."""
from __future__ import annotations

import torch


def batch_from_schema(agent, fields):
    """Adapt the shared batch layout to recurrent policy calculations."""
    space = agent.action_space
    obs = fields["obs"].transpose(0, 1)
    actions = fields["actions"].transpose(0, 1).float()
    last_actions = fields["last_actions"].transpose(0, 1)
    valid = fields["valid"].transpose(0, 1)
    active = fields["active"].transpose(0, 1).clone()
    steps, batch_size, n_agents, _ = actions.shape
    lengths = fields["valid"].sum(dim=1).long()
    terminated = fields["terminated"].transpose(0, 1) | ~valid.unsqueeze(-1)
    truncated = fields["truncated"].transpose(0, 1)
    for row, length in enumerate(lengths.tolist()):
        active[length, row] = ~terminated[length - 1, row]

    previous_dim = space.layout["previous_action_dim"]
    previous = torch.zeros((steps + 1, batch_size, n_agents, previous_dim),
                           dtype=torch.float32, device=obs.device)
    has_previous = (~fields["starts_with_episode"]).unsqueeze(-1).expand(-1, n_agents).clone()

    def encode_previous(values):
        if space.layout["category_counts"]:
            return torch.cat([
                torch.nn.functional.one_hot(values[..., index].long(), count)
                for index, count in enumerate(space.layout["category_counts"])
            ], dim=-1).float()
        return values.float()

    for time in range(steps):
        previous[time] = (
            encode_previous(last_actions[time])
            * (has_previous & valid[time].unsqueeze(-1)).unsqueeze(-1)
        )
        has_previous |= active[time]
    for row, length in enumerate(lengths.tolist()):
        last = torch.where(
            active[length - 1, row].unsqueeze(-1),
            actions[length - 1, row], last_actions[length - 1, row],
        )
        previous[length, row] = encode_previous(last) * has_previous[row].unsqueeze(-1)

    states = fields["state"].transpose(0, 1).unsqueeze(2).expand(-1, -1, n_agents, -1)
    parts = [obs]
    if agent.config.obs_last_action:
        parts.append(previous)
    if agent.config.obs_agent_id:
        parts.append(torch.eye(n_agents, device=obs.device).expand(steps + 1, batch_size, n_agents, n_agents))
    masks = fields["action_masks"]
    if masks is not None:
        masks = masks[:, :-1].transpose(0, 1).clone()
        masks = torch.where(active[:-1].unsqueeze(-1), masks, torch.ones_like(masks))
    if fields["old_log_probs"] is None:
        raise ValueError("MAPPO requires recorded behavior log probabilities; resample with policy_rnn")
    temperature = fields["temperature"]
    if temperature is None:
        temperature = torch.ones_like(fields["old_log_probs"])
    hidden = fields["hidden_in"]
    if hidden is None:
        hidden = torch.zeros((batch_size, n_agents, agent.config.num_layers, agent.config.hidden_dim),
                             dtype=torch.float32, device=obs.device)
    prefixes = [
        None if prefix is None else batch_from_schema(agent, {**prefix, "burn_in": [None]})
        for prefix in fields.get("burn_in", [None] * batch_size)
    ]
    return {
        "inputs": torch.cat(parts, -1),
        "obs": obs,
        "previous": previous,
        "states": states,
        "actions": actions,
        "valid": valid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_agents, 1).float(),
        "active": active.unsqueeze(-1).float(),
        "rewards": fields["rewards"].transpose(0, 1).unsqueeze(-1),
        "terminated": terminated.unsqueeze(-1).float(),
        "truncated": truncated.unsqueeze(-1).float(),
        "old_log_probs": fields["old_log_probs"].transpose(0, 1).unsqueeze(-1),
        "temperature": temperature.transpose(0, 1).unsqueeze(-1),
        "action_masks": masks,
        "hidden": hidden,
        "burn_in": prefixes,
    }


@torch.no_grad()
def burn_in_hidden(actor, prefix):
    """只消费历史动作前的输入，不额外消费窗口首帧。支持AEC未行动槽保留记忆。"""
    inputs = prefix['inputs'][:-1, 0]
    active = prefix['active'][:-1, 0]
    steps, agents, dim = inputs.shape
    hidden = prefix['hidden'][0].clone()
    if torch.all(active > 0):
        _, hidden = actor(inputs.reshape(steps*agents, dim), hidden,
                          torch.ones(steps*agents, 1, device=inputs.device))
    else:
        for k in range(steps):
            _, next_hidden = actor(inputs[k], hidden, torch.ones(agents, 1, device=inputs.device))
            hidden = torch.where(active[k].reshape(agents,1,1).bool(), next_hidden, hidden)
    return hidden.detach()


def actor_outputs(actor, batch, episode_indices=None):
    """Unroll sampled windows with current-policy burn-in; inactive AEC slots retain memory."""
    x = batch['inputs'] if episode_indices is None else batch['inputs'][:,episode_indices]
    active = batch['active'] if episode_indices is None else batch['active'][:,episode_indices]
    hidden = batch['hidden'] if episode_indices is None else batch['hidden'][episode_indices]
    t,b,n,d = x.shape
    hidden = hidden.clone()
    indices = range(b) if episode_indices is None else episode_indices.tolist()
    # 每个PPO epoch/minibatch都用当前Actor重建记忆；前缀不进入loss与新样本步数。
    for row, index in enumerate(indices):
        prefix = batch['burn_in'][index]
        if prefix is not None:
            hidden[row] = burn_in_hidden(actor, prefix)
    hidden = hidden.reshape(b*n, actor.num_layers, actor.hidden_dim)
    # Parallel episodes, including padding: use upstream's efficient full GRU unroll.
    valid = batch['valid'] if episode_indices is None else batch['valid'][:,episode_indices]
    if torch.all(active[:-1] >= valid):
        outputs, _ = actor(x.reshape(t*b*n,d), hidden, torch.ones(t*b*n,1,device=x.device))
        return outputs.reshape(t,b,n,-1)
    rows = []
    for k in range(t):
        out, next_hidden = actor(x[k].reshape(b*n,d), hidden, torch.ones(b*n,1,device=x.device))
        running = active[k].reshape(b*n,1,1).bool()
        hidden = torch.where(running, next_hidden, hidden)
        rows.append(out.reshape(b,n,-1))
    return torch.stack(rows)
