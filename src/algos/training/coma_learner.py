"""Counterfactual policy gradients for discrete and bounded continuous actions."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from algos.learners import Learner
from algos.modules.critics.coma import JointActionCritic
from algos.utils.normalization import RunningMoments
from algos.utils.policy_batch import actor_outputs, batch_from_schema
from common.plugins import learner_plugin
from config import ComaConfig
from algos.learners.batch_schema import DefaultEpisodeFields
from sample.model import EpisodeBatch
from utils.checkpoint import extract_state_dict
from utils.dim_tools import state_shape


@learner_plugin(ComaConfig)
class COMALearner(Learner[DefaultEpisodeFields]):
    """Shared recurrent policy with a counterfactual centralized critic."""

    supported_action_spaces = ("discrete", "continuous", "multi_continuous")

    def __init__(self, algo_type, config, agent, env, device, net_source="blanco",
                 model_file_path=None, caller=None, uploader=None, **kwargs):
        if agent.agent_type not in ("policy_rnn", "mappo_rnn"):
            raise ValueError("COMA requires a policy_rnn agent")
        if not config.common_reward:
            raise ValueError("COMA requires a shared team reward")
        self._validate_config(config.algo_config)
        self.update_cnt = 0
        super().__init__(algo_type, config, agent, env, device, net_source,
                         model_file_path, caller, uploader, **kwargs)

    @property
    def agent_net(self):
        return self.actor

    @property
    def models(self):
        return {"agent_net": self.actor, "critic_net": self.critic}

    def build_empty_network(self, learner_config, **kwargs):
        cfg = learner_config.algo_config
        self.actor = copy.deepcopy(self.agent.agent_net).to(self.device)
        group_shape = learner_config.state_shape if self.agent.config.agent_groups else state_shape(self.env)
        state_dim = int(np.prod(group_shape))
        layout = self.agent.action_space.layout
        self.critic_action_dim = (
            layout["category_counts"][0] if layout["kind"] == "discrete" else layout["value_dim"]
        )
        self.critic = JointActionCritic(
            state_dim, self.agent.n_agents, self.critic_action_dim, self.agent.config.hidden_dim,
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.reward_moments = RunningMoments().to(self.device)
        self.return_moments = RunningMoments().to(self.device)
        return self.actor, self.critic

    def build_network_from_checkpoint(self, checkpoint, learner_config, **kwargs):
        self.build_empty_network(learner_config, **kwargs)
        actor_state = extract_state_dict(checkpoint["agent_net"])
        critic_state = extract_state_dict(checkpoint["critic_net"])
        self.actor.load_state_dict(actor_state, strict=True)
        self.critic.load_state_dict(critic_state, strict=True)
        self.target_critic.load_state_dict(self.critic.state_dict())
        return self.actor, self.critic

    def _offline_net_weights(self):
        return {"critic_net": self.critic.state_dict()}

    def snapshot_state(self):
        return {
            "agent_net": self.actor.state_dict(),
            "critic_net": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "reward_moments": self.reward_moments.state_dict(),
            "return_moments": self.return_moments.state_dict(),
            "update_cnt": self.update_cnt,
        }

    def restore_state(self, state):
        self.actor.load_state_dict(state["agent_net"])
        self.critic.load_state_dict(state["critic_net"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.reward_moments.load_state_dict(state["reward_moments"])
        self.return_moments.load_state_dict(state["return_moments"])
        self.update_cnt = state["update_cnt"]

    def dynamic_modify_config(self, updates):
        candidate = copy.copy(self.config.algo_config)
        if isinstance(updates, dict):
            for name in ("learn_times_per_episode", "q_nstep", "counterfactual_samples"):
                if name in updates:
                    setattr(candidate, name, updates[name])
        self._validate_config(candidate)
        super().dynamic_modify_config(updates)
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = self.config.algo_config.lr

    @staticmethod
    def _validate_config(config):
        if config.learn_times_per_episode != 1:
            raise ValueError("COMA requires learn_times_per_episode=1 for on-policy samples")
        for name in ("q_nstep", "counterfactual_samples"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"COMA {name} must be a positive integer")

    def _encode_actions(self, actions):
        if self.agent.action_space.kind == "discrete":
            return F.one_hot(actions[..., 0].long(), self.critic_action_dim).float()
        return actions.float()

    @torch.no_grad()
    def _counterfactual_baseline(self, state, joint_actions, active, distribution):
        baselines = []
        n_agents = joint_actions.shape[-2]
        for agent_index in range(n_agents):
            if self.agent.action_space.kind == "discrete":
                probs = distribution.categoricals[0].probs[..., agent_index, :]
                baseline = torch.zeros_like(probs[..., 0])
                for action_index in range(self.critic_action_dim):
                    alternative = joint_actions.clone()
                    alternative[..., agent_index, :] = F.one_hot(
                        torch.full_like(probs[..., 0], action_index, dtype=torch.long),
                        self.critic_action_dim,
                    ).float()
                    baseline += probs[..., action_index] * self.critic(state, alternative, active)
            else:
                baseline = torch.zeros_like(joint_actions[..., 0, 0])
                for _ in range(self.config.algo_config.counterfactual_samples):
                    sampled = distribution.sample()
                    alternative = joint_actions.clone()
                    alternative[..., agent_index, :] = sampled[..., agent_index, :]
                    baseline += self.critic(state, alternative, active)
                baseline /= self.config.algo_config.counterfactual_samples
            baselines.append(baseline)
        return torch.stack(baselines, dim=-1)

    def _nstep_targets(self, rewards, terminal, truncated, valid, next_values):
        steps = rewards.shape[0]
        targets = torch.zeros_like(rewards)
        horizon = self.config.algo_config.q_nstep
        gamma = self.config.gamma
        for start in range(steps):
            remaining = valid[start].float()
            for offset in range(horizon):
                time = start + offset
                if time >= steps:
                    break
                present = remaining * valid[time].float()
                targets[start] += (gamma ** offset) * present * rewards[time]
                continuing = present * (~terminal[time]).float()
                next_valid = valid[time + 1] if time + 1 < steps else torch.zeros_like(valid[time])
                stop = truncated[time] | ~next_valid
                if offset == horizon - 1:
                    stop = torch.ones_like(stop)
                targets[start] += (gamma ** (offset + 1)) * continuing * stop.float() * next_values[time]
                remaining = continuing * (~stop).float()
        return targets

    @staticmethod
    def _team_terminal(terminated, truncated, no_next_actor):
        # A previously terminated agent may no longer receive a termination
        # flag on the step where the last active agent finishes. Replay also
        # records when the joint trajectory has no next actor; a time limit must
        # still bootstrap the surviving agents from the final observation.
        return terminated.all(-1) | (no_next_actor & ~truncated.any(-1))

    def update(self, batch: EpisodeBatch[DefaultEpisodeFields]):
        fields = batch.data
        batch = batch_from_schema(self.agent, fields)
        policy_mask = (batch["valid"] * batch["active"][:-1]).squeeze(-1).bool()
        if not policy_mask.any():
            raise ValueError("COMA batch has no active decisions")
        state = fields["state"].transpose(0, 1)
        active = fields["active"].transpose(0, 1)
        actions = batch["actions"]
        joint_actions = self._encode_actions(actions)
        outputs = actor_outputs(self.actor, batch)
        distribution = self.actor.distribution(outputs[:-1], batch["action_masks"], batch["temperature"])
        logs = distribution.log_prob(actions).squeeze(-1)
        recorded = batch["old_log_probs"].squeeze(-1)
        if not torch.allclose(logs[policy_mask].detach(), recorded[policy_mask], atol=1e-5, rtol=1e-5):
            raise ValueError("COMA requires trajectories collected by the current policy; resample stale data")

        with torch.no_grad():
            q_taken = self.critic(state[:-1], joint_actions, active[:-1])
            baseline = self._counterfactual_baseline(state[:-1], joint_actions, active[:-1], distribution)
            advantage = q_taken.unsqueeze(-1) - baseline

            valid = fields["valid"].transpose(0, 1)
            next_valid = torch.cat((valid[1:], torch.zeros_like(valid[:1])), dim=0)
            terminated = fields["terminated"].transpose(0, 1)
            truncated = fields["truncated"].transpose(0, 1)
            bootstrap_active = active[1:].clone()
            time_limit = truncated.any(-1) & ~next_valid
            bootstrap_active = torch.where(time_limit.unsqueeze(-1), ~terminated, bootstrap_active)
            masks = fields["action_masks"]
            if masks is not None:
                masks = masks.transpose(0, 1)
                masks = torch.where(active.unsqueeze(-1), masks, torch.ones_like(masks))
                masks = torch.where(masks.any(-1, keepdim=True), masks, torch.ones_like(masks))
            temperatures = torch.cat((batch["temperature"], torch.ones_like(batch["temperature"][:1])), dim=0)
            next_distribution = self.actor.distribution(outputs.detach(), masks, temperatures)
            sampled = self._encode_actions(next_distribution.sample())
            recorded_next = torch.cat((joint_actions[1:], sampled[-1:]), dim=0)
            next_actions = torch.where(
                next_valid.unsqueeze(-1).unsqueeze(-1), recorded_next, sampled[1:],
            )
            target_values = self.target_critic(state[1:], next_actions, bootstrap_active)
            rewards = fields["team_rewards"].transpose(0, 1)
            if self.config.standardise_rewards:
                self.reward_moments.update(rewards[valid])
                rewards = self.reward_moments.normalize(rewards)
            if self.config.standardise_returns:
                target_values = self.return_moments.denormalize(target_values)
            team_terminal = self._team_terminal(
                terminated, truncated, fields["team_terminated"].transpose(0, 1),
            )
            team_truncated = truncated.all(-1)
            targets = self._nstep_targets(rewards, team_terminal, team_truncated, valid, target_values)
            if self.config.standardise_returns:
                self.return_moments.update(targets[valid])
                targets = self.return_moments.normalize(targets)

        q_values = self.critic(state[:-1], joint_actions, active[:-1])
        critic_loss = ((q_values - targets).square() * valid).sum() / valid.sum().clamp_min(1)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_norm_clip)
        self.critic_optimizer.step()

        count = policy_mask.sum().clamp_min(1)
        entropy = (distribution.entropy().squeeze(-1) * policy_mask).sum() / count
        policy_loss = -(logs * advantage.detach() * policy_mask).sum() / count
        actor_loss = policy_loss - self.config.algo_config.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_norm_clip)
        self.actor_optimizer.step()

        self.update_cnt += 1
        interval = self.config.algo_config.target_update_interval_or_tau
        if interval > 1 and self.update_cnt % int(interval) == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        elif interval <= 1:
            for target, current in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.data.copy_(target.data * (1 - interval) + current.data * interval)
        return {
            "policy_loss": float(policy_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "entropy": float(entropy.detach()),
            "agent_grad_norm": float(actor_grad),
            "critic_grad_norm": float(critic_grad),
            "counterfactual_advantage": float(advantage[policy_mask].mean()),
        }
