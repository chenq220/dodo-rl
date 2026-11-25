"""Off-policy multi-agent deterministic policy gradients."""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from algos.learners import Learner
from algos.modules.agents.maddpg_agent import validated_actor_state_dict
from algos.modules.critics.maddpg import build_critics
from algos.utils.normalization import RunningMoments
from common.plugins import learner_plugin
from config import MaddpgConfig
from envs.base import AECEnvironment
from envs.action.spaces import ActionSpace
from algos.learners.batch_schema import MADDPG_EPISODE_SCHEMA, MaddpgEpisodeFields
from sample.model import EpisodeBatch, SampledEpisodes
from utils.checkpoint import validated_checkpoint_state_dict


@learner_plugin(MaddpgConfig)
class MADDPGLearner(Learner[MaddpgEpisodeFields]):
    """One deterministic actor and one centralized Q function per agent slot."""

    supported_action_spaces = ("continuous", "multi_continuous")
    # 前馈 TD 不需要 burn-in / 集中状态 / 旧策略统计，按最小 schema 组装和传输。
    batch_schema = MADDPG_EPISODE_SCHEMA

    def __init__(self, algo_type, config, agent, env, device, net_source="blanco",
                 model_file_path=None, caller=None, uploader=None, **kwargs):
        if agent.agent_type != "maddpg_mlp":
            raise ValueError("MADDPG requires a maddpg_mlp agent")
        if isinstance(env, AECEnvironment):
            raise ValueError("MADDPG requires Parallel joint transitions")
        self._validate_config(config.algo_config)
        self.update_cnt = 0
        super().__init__(algo_type, config, agent, env, device, net_source,
                         model_file_path, caller, uploader, **kwargs)

    def learn_update(self, batch_data, generation=0):
        if generation > self.generation:
            self.restore_models_snapshot()
            self.generation = generation
        strategy = batch_data.sample_strategy if isinstance(batch_data, SampledEpisodes) else None
        return self.update(self.assemble_batch(batch_data, sample_strategy=strategy))

    def assemble_batch(self, episodes, *, sample_strategy: str | None = None):
        """MADDPG consumes replay's normalized actions directly."""
        if not episodes or any(episode.seq_len <= 0 for episode in episodes):
            raise ValueError("MADDPG batch requires nonempty trajectories")
        if any(type(episode) is not type(episodes[0]) for episode in episodes):
            raise TypeError(f"Learner {self.name} received incompatible trajectory types")
        space = self.agent.action_space
        signature = space.signature
        batch_size = len(episodes)
        steps = max(episode.seq_len for episode in episodes)
        n_agents = self.agent.n_agents
        obs_dim = self.agent.config.obs_dim
        action_dim = space.layout["value_dim"]
        shape = (batch_size, steps, n_agents)
        arrays = {
            "obs": np.zeros((batch_size, steps + 1, n_agents, obs_dim), np.float32),
            "actions": np.zeros((*shape, action_dim), np.float32),
            "last_actions": np.zeros((*shape, action_dim), np.float32),
            "rewards": np.zeros(shape, np.float32),
            "team_rewards": np.zeros((batch_size, steps), np.float32),
            "terminated": np.zeros(shape, bool),
            "truncated": np.zeros(shape, bool),
            "valid": np.zeros((batch_size, steps), bool),
            "active": np.zeros((batch_size, steps + 1, n_agents), bool),
        }
        for b, episode in enumerate(episodes):
            ids = tuple(episode.agent_ids) or tuple(self.agent.agent_list)
            if len(ids) != n_agents:
                raise ValueError(f"trajectory {episode.episode_num} has incorrect agent count")
            episode.validate_activity_masks(ids)
            for aid, spec in episode.action_space_specs.items():
                if aid in ids and ActionSpace.from_dict(spec).signature != signature:
                    raise ValueError("replay action layout does not match learner action space")
            length = episode.seq_len
            arrays["valid"][b, :length] = True
            arrays["team_rewards"][b, :length] = episode.step_rewards[:length]
            for n, aid in enumerate(ids):
                arrays["obs"][b, :length, n] = np.asarray(episode.obs[aid]).reshape(length, obs_dim)
                arrays["obs"][b, length, n] = np.asarray(episode.next_obs[aid][-1]).reshape(obs_dim)
                for name, field in (("rewards", episode.rewards), ("terminated", episode.terminations),
                                    ("truncated", episode.truncations)):
                    arrays[name][b, :length, n] = field[aid][:length]
                active = episode.active_agent_mask.get(aid)
                arrays["active"][b, :length, n] = active[:length] if active is not None else True
                following = episode.next_active_agent_mask.get(aid)
                if active is not None and following is not None and len(following) >= length:
                    arrays["active"][b, length, n] = following[length - 1]
                elif active is not None:
                    metadata = episode.metadata[length - 1] or {} if length <= len(episode.metadata) else {}
                    next_id = metadata.get("next_active_agent_id")
                    available = (episode.avail_actions_next[length - 1].get(aid)
                                 if length <= len(episode.avail_actions_next) else None)
                    arrays["active"][b, length, n] = (
                        aid == next_id if next_id is not None else available is not None and bool(any(available))
                    )
                for name, field in (("actions", episode.actions), ("last_actions", episode.last_actions)):
                    for t, record in enumerate(field[aid][:length]):
                        if record is None:
                            continue
                        if isinstance(record, dict):
                            if record.get("format_version") != 1 or record.get("space_signature") != signature:
                                raise ValueError("action record space layout/signature mismatch")
                            value = record["value"]
                        else:
                            value = space.encode(space.decode(record))
                        arrays[name][b, t, n] = np.asarray(value).reshape(action_dim)
        return EpisodeBatch(
            {name: torch.as_tensor(value, device=self.device) for name, value in arrays.items()},
            sample_strategy=sample_strategy,
        )

    @staticmethod
    def _validate_config(config):
        update = config.target_update_interval_or_tau
        if not math.isfinite(update) or update <= 0 or (update > 1 and update != int(update)):
            raise ValueError("MADDPG target_update_interval_or_tau must be in (0, 1] or an integer interval > 1")
        if not math.isfinite(config.lr) or config.lr <= 0:
            raise ValueError("MADDPG lr must be finite and positive")
        for name in ("actor_lr", "critic_lr"):
            value = getattr(config, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"MADDPG {name} must be finite and positive or None")
        warmup = config.actor_warmup_updates
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("MADDPG actor_warmup_updates must be a nonnegative integer")
        if isinstance(config.n_step, bool) or not isinstance(config.n_step, int) or config.n_step < 1:
            raise ValueError("MADDPG n_step must be a positive integer")
        for name in ("bc_anchor_coef", "bc_anchor_final_coef"):
            value = getattr(config, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"MADDPG {name} must be finite and nonnegative")
        decay = config.bc_anchor_decay_updates
        if isinstance(decay, bool) or not isinstance(decay, int) or decay < 0:
            raise ValueError("MADDPG bc_anchor_decay_updates must be a nonnegative integer")
        if not math.isfinite(config.reg) or config.reg < 0:
            raise ValueError("MADDPG reg must be finite and nonnegative")

    @staticmethod
    def _actor_lr(config):
        return float(config.actor_lr if config.actor_lr is not None else config.lr)

    @staticmethod
    def _critic_lr(config):
        return float(config.critic_lr if config.critic_lr is not None else config.lr)

    @property
    def agent_net(self):
        return self.actor

    @property
    def models(self):
        return {"agent_net": self.actor, "critic_net": self.critic}

    def build_empty_network(self, learner_config, **kwargs):
        config = learner_config.algo_config
        n_agents = self.agent.n_agents
        critic_input_dim = n_agents * (self.agent.config.obs_dim + self.agent.action_layout["value_dim"] + 1)
        self.actor = copy.deepcopy(self.agent.agent_net).to(self.device)
        self.critic = build_critics(n_agents, critic_input_dim, config.hidden_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.bc_anchor_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.bc_anchor_actor.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr(config))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._critic_lr(config))
        self.reward_moments = nn.ModuleList(RunningMoments() for _ in range(n_agents)).to(self.device)
        self.return_moments = nn.ModuleList(RunningMoments() for _ in range(n_agents)).to(self.device)
        return self.actor, self.critic

    def build_network_from_checkpoint(self, checkpoint, learner_config, **kwargs):
        self.build_empty_network(learner_config, **kwargs)
        self.actor.load_state_dict(
            validated_actor_state_dict(self.actor, checkpoint.get("agent_net"), agent_name=self.name),
        )
        self.critic.load_state_dict(validated_checkpoint_state_dict(
            self.critic, checkpoint.get("critic_net"), agent_name=self.name, model_name="critic_net",
            expected_dimensions={"n_agents": self.agent.n_agents},
        ))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.bc_anchor_actor.load_state_dict(self.actor.state_dict())
        return self.actor, self.critic

    def _offline_net_weights(self):
        return {"critic_net": self.critic.state_dict()}

    def snapshot_state(self):
        return {
            "agent_net": self.actor.state_dict(),
            "critic_net": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "bc_anchor_actor": self.bc_anchor_actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "reward_moments": self.reward_moments.state_dict(),
            "return_moments": self.return_moments.state_dict(),
            "update_cnt": self.update_cnt,
        }

    def restore_state(self, state):
        self.actor.load_state_dict(state["agent_net"])
        self.critic.load_state_dict(state["critic_net"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_critic.load_state_dict(state["target_critic"])
        if "bc_anchor_actor" in state:
            self.bc_anchor_actor.load_state_dict(state["bc_anchor_actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.reward_moments.load_state_dict(state["reward_moments"])
        self.return_moments.load_state_dict(state["return_moments"])
        self.update_cnt = state["update_cnt"]

    def dynamic_modify_config(self, updates):
        if isinstance(updates, dict):
            candidate = replace(self.config.algo_config, **updates)
            self._validate_config(candidate)
            if candidate.hidden_dim != self.config.algo_config.hidden_dim:
                raise ValueError("MADDPG hidden_dim cannot change after network construction")
            if candidate.agent_output_type != self.agent.fixed_output_type:
                raise ValueError("MADDPG requires continuous_action actor outputs")
        super().dynamic_modify_config(updates)
        config = self.config.algo_config
        for optimizer, lr in (
                (self.actor_optimizer, self._actor_lr(config)),
                (self.critic_optimizer, self._critic_lr(config)),
        ):
            for group in optimizer.param_groups:
                group["lr"] = lr

    def _actor_inputs(self, obs, last_actions):
        parts = [obs]
        if self.agent.config.obs_last_action:
            parts.append(last_actions)
        if self.agent.config.obs_agent_id:
            identity = torch.eye(self.agent.n_agents, device=obs.device)
            parts.append(identity.expand(obs.shape[0], -1, -1))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _joint_inputs(obs, actions, active):
        mask = active.unsqueeze(-1)
        return torch.cat((
            obs.masked_fill(~mask, 0).flatten(1),
            actions.masked_fill(~mask, 0).flatten(1),
            active.to(obs.dtype),
        ), dim=-1)

    @staticmethod
    def _q_values(critics, inputs):
        return torch.cat([critic(inputs) for critic in critics], dim=-1)

    @torch.no_grad()
    def _targets(self, rewards, next_obs, last_actions, active, next_active):
        next_mean, _ = self.target_actor(self._actor_inputs(next_obs, last_actions))
        next_policy = self.target_actor.deterministic_actions(next_mean)
        next_inputs = self._joint_inputs(next_obs, next_policy, next_active)
        next_values = self._q_values(self.target_critic, next_inputs)
        targets = []
        for slot, (reward_moments, return_moments) in enumerate(zip(self.reward_moments, self.return_moments)):
            reward = rewards[:, slot]
            future = next_values[:, slot]
            if self.config.standardise_rewards:
                reward_moments.update(reward[active[:, slot]])
                reward = reward_moments.normalize(reward)
            if self.config.standardise_returns:
                future = return_moments.denormalize(future)
            target = reward + self.config.gamma * future.masked_fill(~next_active[:, slot], 0)
            if self.config.standardise_returns:
                return_moments.update(target[active[:, slot]])
                target = return_moments.normalize(target)
            targets.append(target)
        return torch.stack(targets, dim=-1)

    @torch.no_grad()
    def _nstep_targets(self, fields, learn_active):
        """为每个窗口起点计算不跨 episode 边界的 n-step TD 目标。"""
        valid = fields["valid"]
        active = fields["active"][:, :-1]
        next_active = (
            fields["active"][:, 1:] | (active & fields["truncated"])
        ) & ~fields["terminated"]
        rewards = fields["rewards"]
        if self.config.common_reward:
            rewards = fields["team_rewards"].unsqueeze(-1).expand_as(rewards)

        normalized_rewards = rewards.clone()
        next_previous = torch.where(
            active.unsqueeze(-1), fields["actions"], fields["last_actions"],
        )
        batch_size, sequence_length, n_agents = rewards.shape
        flat_next_obs = fields["obs"][:, 1:].reshape(
            batch_size * sequence_length, n_agents, -1,
        )
        flat_previous = next_previous.reshape(
            batch_size * sequence_length, n_agents, -1,
        )
        flat_next_active = next_active.reshape(batch_size * sequence_length, n_agents)
        next_mean, _ = self.target_actor(self._actor_inputs(flat_next_obs, flat_previous))
        next_policy = self.target_actor.deterministic_actions(next_mean)
        next_inputs = self._joint_inputs(flat_next_obs, next_policy, flat_next_active)
        next_values = self._q_values(self.target_critic, next_inputs).reshape(
            batch_size, sequence_length, n_agents,
        )

        for slot, (reward_moments, return_moments) in enumerate(
                zip(self.reward_moments, self.return_moments)):
            reward_mask = valid & active[:, :, slot]
            if self.config.standardise_rewards:
                reward_moments.update(rewards[:, :, slot][reward_mask])
                normalized_rewards[:, :, slot] = reward_moments.normalize(rewards[:, :, slot])
            if self.config.standardise_returns:
                next_values[:, :, slot] = return_moments.denormalize(next_values[:, :, slot])

        targets = torch.zeros_like(rewards)
        alive = valid.unsqueeze(-1) & active
        gamma = float(self.config.gamma)
        horizon = self.config.algo_config.n_step
        for offset in range(min(horizon, sequence_length)):
            remaining = sequence_length - offset
            current = (
                alive[:, :remaining]
                & valid[:, offset:].unsqueeze(-1)
                & active[:, offset:]
            )
            targets[:, :remaining] += (
                (gamma ** offset) * normalized_rewards[:, offset:] * current
            )

            can_continue = torch.zeros_like(current)
            if offset + 1 < horizon and remaining > 1:
                can_continue[:, :-1] = (
                    current[:, :-1]
                    & next_active[:, offset:-1]
                    & ~fields["truncated"][:, offset:-1]
                    & valid[:, offset + 1:].unsqueeze(-1)
                )
            bootstrap = current & ~can_continue & next_active[:, offset:]
            targets[:, :remaining] += (
                (gamma ** (offset + 1)) * next_values[:, offset:] * bootstrap
            )
            alive[:, :remaining] = can_continue

        if self.config.standardise_returns:
            for slot, return_moments in enumerate(self.return_moments):
                mask = learn_active[:, :, slot]
                return_moments.update(targets[:, :, slot][mask])
                targets[:, :, slot] = return_moments.normalize(targets[:, :, slot])
        return targets

    def _bc_anchor_coefficient(self):
        config = self.config.algo_config
        if config.bc_anchor_decay_updates == 0:
            return float(config.bc_anchor_coef)
        actor_updates = max(0, self.update_cnt - config.actor_warmup_updates)
        progress = min(1.0, actor_updates / config.bc_anchor_decay_updates)
        return float(
            config.bc_anchor_coef
            + progress * (config.bc_anchor_final_coef - config.bc_anchor_coef)
        )

    @torch.no_grad()
    def _update_targets(self):
        tau = self.config.algo_config.target_update_interval_or_tau
        if tau > 1:
            if self.update_cnt % int(tau):
                return
            tau = 1.0
        for target, online in ((self.target_actor, self.actor), (self.target_critic, self.critic)):
            for target_parameter, parameter in zip(target.parameters(), online.parameters()):
                target_parameter.lerp_(parameter, tau)

    def update(self, batch: EpisodeBatch[MaddpgEpisodeFields]):
        fields = batch.data
        valid = fields["valid"] & fields["active"][:, :-1].any(dim=-1)
        if (self.config.algo_config.n_step > 1
                and batch.sample_strategy == "uniform_steps"):
            starts = torch.zeros_like(valid)
            starts[:, 0] = valid[:, 0]
            valid = starts
        if not valid.any():
            raise ValueError("MADDPG batch has no active decisions")
        obs = fields["obs"][:, :-1][valid]
        next_obs = fields["obs"][:, 1:][valid]
        actions = fields["actions"][valid]
        last_actions = fields["last_actions"][valid]
        active = fields["active"][:, :-1][valid]
        terminated = fields["terminated"][valid]
        truncated = fields["truncated"][valid]
        next_active = (fields["active"][:, 1:][valid] | (active & truncated)) & ~terminated
        rewards = fields["rewards"][valid]
        if self.config.common_reward:
            rewards = fields["team_rewards"][valid].unsqueeze(-1).expand_as(rewards)

        next_previous = torch.where(active.unsqueeze(-1), actions, last_actions)
        if self.config.algo_config.n_step == 1:
            targets = self._targets(rewards, next_obs, next_previous, active, next_active)
        else:
            learn_active = valid.unsqueeze(-1) & fields["active"][:, :-1]
            targets = self._nstep_targets(fields, learn_active)[valid]
        values = self._q_values(self.critic, self._joint_inputs(obs, actions, active))
        td_error = values[active] - targets[active]
        critic_loss = td_error.square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_norm_clip)
        self.critic_optimizer.step()

        actor_warmup = self.update_cnt < self.config.algo_config.actor_warmup_updates
        policy_loss_value = 0.0
        actor_grad = 0.0
        anchor_loss_value = 0.0
        anchor_coefficient = 0.0
        if not actor_warmup:
            self.critic.requires_grad_(False)
            try:
                mean, _ = self.actor(self._actor_inputs(obs, last_actions))
                policy_actions = self.actor.deterministic_actions(mean)
                policy_values = []
                for slot, critic in enumerate(self.critic):
                    joint = torch.cat((actions[:, :slot], policy_actions[:, slot:slot + 1],
                                       actions[:, slot + 1:]), dim=1)
                    policy_values.append(critic(self._joint_inputs(obs, joint, active)))
                policy_loss = -torch.cat(policy_values, dim=-1)[active].mean()
                anchor_coefficient = self._bc_anchor_coefficient()
                if anchor_coefficient:
                    with torch.no_grad():
                        anchor_mean, _ = self.bc_anchor_actor(self._actor_inputs(obs, last_actions))
                        anchor_actions = self.bc_anchor_actor.deterministic_actions(anchor_mean)
                    anchor_loss = (policy_actions[active] - anchor_actions[active]).square().mean()
                else:
                    anchor_loss = policy_loss.new_zeros(())
                actor_loss = (
                    policy_loss
                    + self.config.algo_config.reg * mean[active].square().mean()
                    + anchor_coefficient * anchor_loss
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad = float(nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.grad_norm_clip,
                ))
                self.actor_optimizer.step()
                policy_loss_value = float(policy_loss.detach())
                anchor_loss_value = float(anchor_loss.detach())
            finally:
                self.critic.requires_grad_(True)

        self.update_cnt += 1
        self._update_targets()
        return {
            "critic_loss": float(critic_loss.detach()),
            "critic_grad_norm": float(critic_grad),
            "td_error_abs": float(td_error.detach().abs().mean()),
            "q_taken_mean": float(values.detach()[active].mean()),
            "target_mean": float(targets[active].mean()),
            "pg_loss": policy_loss_value,
            "agent_grad_norm": actor_grad,
            "actor_warmup": float(actor_warmup),
            "bc_anchor_loss": anchor_loss_value,
            "bc_anchor_coef": anchor_coefficient,
            "n_step": float(self.config.algo_config.n_step),
        }
