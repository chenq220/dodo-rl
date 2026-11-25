"""Legacy MAPPO trajectory conversion and generalized advantage estimation."""
from __future__ import annotations

import numpy as np
import torch

from algos.utils.action_masks import action_mask_vector
from envs.action.spaces import ActionSpace


def recorded_action(space, value):
    if value is None:
        return None
    if isinstance(value, dict):
        return ActionSpace.restore(value, expected_space=space)
    return space.decode(value)


def build_batch(agent, episodes, device):
    episodes = [ep for ep in episodes if ep.seq_len]
    if not episodes:
        raise ValueError('MAPPO requires nonempty episodes')
    space, ids = agent.action_space, agent.agent_list
    b, t, n = len(episodes), max(ep.seq_len for ep in episodes), len(ids)
    act_dim = space.layout['value_dim']
    obs_dim = agent.dim_info.value
    state_dim = np.asarray(episodes[0].states[0]).size
    arrays = {
        'inputs': np.zeros((t+1,b,n,obs_dim), np.float32),
        'obs': np.zeros((t+1,b,n,agent.config.obs_dim), np.float32),
        'previous': np.zeros((t+1,b,n,space.layout['previous_action_dim']), np.float32),
        'states': np.zeros((t+1,b,n,state_dim), np.float32),
        'actions': np.zeros((t,b,n,act_dim), np.float32),
        'valid': np.zeros((t,b,n,1), np.float32),
        'active': np.zeros((t+1,b,n,1), np.float32),
        'rewards': np.zeros((t,b,n,1), np.float32),
        'terminated': np.ones((t,b,n,1), np.float32),
        'truncated': np.zeros((t,b,n,1), np.float32),
        'old_log_probs': np.zeros((t,b,n,1), np.float32),
        'temperature': np.ones((t,b,n,1), np.float32),
    }
    mask_dim = sum(space.layout['category_counts'])
    if mask_dim:
        arrays['action_masks'] = np.ones((t,b,n,mask_dim), bool)
    hidden = np.zeros((b,n,agent.config.num_layers,agent.config.hidden_dim), np.float32)
    prefixes = []
    for j, ep in enumerate(episodes):
        prefix = ep.burn_in
        if not ep.starts_with_episode or prefix is not None:
            from sample.sample_transfer import EpisodeTransition
            if (not isinstance(prefix, EpisodeTransition) or prefix.seq_len <= 0
                    or not prefix.starts_with_episode
                    or prefix.burn_in is not None
                    or prefix.episode_num != ep.episode_num or tuple(prefix.agent_ids) != tuple(ep.agent_ids)):
                raise ValueError('MAPPO requires complete episodes or same-episode burn_in from reset')
            if (not np.array_equal(prefix.next_states[-1], ep.states[0])
                    or any(not np.array_equal(prefix.next_obs[a][-1], ep.obs[a][0]) for a in ids)
                    or all(prefix.terminations[a][-1] or prefix.truncations[a][-1] for a in ids)):
                raise ValueError('MAPPO burn_in must connect directly to the sampled window without a reset')
            prefixes.append(build_batch(agent, [prefix], device))
        else:
            prefixes.append(None)
        if ep.agent_ids and tuple(ep.agent_ids) != tuple(ids):
            raise ValueError('MAPPO episode agent order does not match policy')
        ep.validate_activity_masks(ids)
        for a, spec in ep.action_space_specs.items():
            if a in ids and ActionSpace.from_dict(spec).signature != space.signature:
                raise ValueError('MAPPO replay action layout does not match policy')
        for i,a in enumerate(ids):
            if a in ep.hidden_in:
                hidden[j,i] = torch.as_tensor(ep.hidden_in[a]).detach().cpu().numpy().reshape(hidden[j,i].shape)
            has_previous = not ep.starts_with_episode
            for k in range(ep.seq_len):
                active = ep.active_agent_mask.get(a, [1.] * ep.seq_len)[k]
                arrays['active'][k,j,i,0] = active
                arrays['valid'][k,j,i,0] = 1
                action = recorded_action(space, ep.actions[a][k])
                if active and action is None:
                    raise ValueError('MAPPO active agent is missing its action')
                if action is not None:
                    arrays['actions'][k,j,i] = np.asarray(space.encode(action)).reshape(-1)
                previous = recorded_action(space, ep.last_actions[a][k]) if has_previous else None
                has_previous = has_previous or bool(active)
                arrays['previous'][k,j,i] = space.encode_previous_action(previous)
                arrays['obs'][k,j,i] = np.asarray(ep.obs[a][k]).flatten()
                arrays['states'][k,j,i] = np.asarray(ep.states[k]).flatten()
                arrays['rewards'][k,j,i,0] = ep.rewards[a][k]
                arrays['terminated'][k,j,i,0] = ep.terminations[a][k]
                arrays['truncated'][k,j,i,0] = ep.truncations[a][k]
                meta = ep.metadata[k] or {}
                logs = meta.get('behavior_log_probs', {})
                if active and a not in logs:
                    raise ValueError('MAPPO requires recorded behavior log probabilities; resample with policy_rnn')
                arrays['old_log_probs'][k,j,i,0] = logs.get(a, 0.)
                arrays['temperature'][k,j,i,0] = logs.get('__temperature__', 1.)
                if mask_dim:
                    masks = ep.avail_actions[k].get(a) if k < len(ep.avail_actions) else None
                    if not active:
                        masks = None
                    arrays['action_masks'][k,j,i] = action_mask_vector(space, masks)
            end = ep.seq_len
            arrays['obs'][end,j,i] = np.asarray(ep.next_obs[a][-1]).flatten()
            arrays['states'][end,j,i] = np.asarray(ep.next_states[-1]).flatten()
            last_value = ep.actions[a][-1] if active else ep.last_actions[a][-1]
            last = recorded_action(space, last_value) if has_previous else None
            arrays['previous'][end,j,i] = space.encode_previous_action(last)
            arrays['active'][end,j,i,0] = 0 if ep.terminations[a][-1] else 1
    parts = [arrays['obs']]
    if agent.config.obs_last_action:
        parts.append(arrays['previous'])
    if agent.config.obs_agent_id:
        parts.append(np.broadcast_to(np.eye(n,dtype=np.float32), (t+1,b,n,n)))
    arrays['inputs'] = np.concatenate(parts, -1)
    tensors = {key: torch.as_tensor(value, device=device) for key,value in arrays.items()}
    tensors['hidden'] = torch.as_tensor(hidden, device=device)
    tensors['burn_in'] = prefixes
    return tensors


def gae(rewards, values, terminated, truncated, valid, gamma, lmbda):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for k in reversed(range(rewards.shape[0])):
        delta = rewards[k] + gamma * (1-terminated[k]) * values[k+1] - values[k]
        carry = (delta + gamma*lmbda*(1-terminated[k])*(1-truncated[k])*carry) * valid[k]
        advantages[k] = carry
    return advantages, advantages + values[:-1]
