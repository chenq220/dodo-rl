"""Native recurrent QMIX learner for discrete actions."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from algos.learners.base import Learner, LOG
from algos.map import AlogsMap
from algos.modules.agents.recurrent_q import IndependentUtilities, RecurrentQNetwork, RecurrentUtility
from algos.modules.mixers.qmix import QMixer
from common.exception import ModelCheckpointCompatibilityException, ReplayBatchShapeException
from common.plugins import learner_plugin
from config import QmixConfig
from algos.learners.batch_schema import DefaultEpisodeFields
from sample.model import EpisodeBatch
from sample.sample_transfer import EpisodeTransition
from utils.checkpoint import validated_checkpoint_state_dict
from utils.dim_tools import state_shape


class RunningStatistics:
    """Running sample moments used by the existing QMIX training contract."""

    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 1e-4

    def update(self, values):
        values = values.reshape(-1, values.size(-1))
        count = values.shape[0]
        mean = values.mean(dim=0)
        var = values.var(dim=0)
        total = self.count + count
        delta = mean - self.mean
        new_mean = self.mean + delta * count / total
        moment = self.var * self.count + var * count + delta.square() * self.count * count / total
        self.mean = new_mean
        self.var = moment / total
        self.count = total


class UtilityController:
    """Build recurrent utility inputs in fixed agent-slot order."""

    def __init__(self, agent):
        self.n_agents = agent.n_agents
        self.shared = agent.agent_type != "rnn_ns"
        self.multilayer = agent.agent_type == "q_rnn"
        self.obs_last_action = agent.config.obs_last_action
        self.obs_agent_id = agent.config.obs_agent_id
        dimensions = agent.dim_info
        use_rnn = agent.config.algo_config.use_rnn
        if self.multilayer:
            self.agent = RecurrentQNetwork(dimensions, use_rnn)
        elif self.shared:
            self.agent = RecurrentUtility(dimensions.value, dimensions.hidden_dim, dimensions.action_dim, use_rnn)
        else:
            self.agent = IndependentUtilities(
                dimensions.value, dimensions.hidden_dim, dimensions.action_dim, self.n_agents, use_rnn,
            )
        self.hidden_states = None

    def init_hidden(self, batch_size):
        if self.multilayer:
            self.hidden_states = self.agent.fc1.weight.new_zeros(
                batch_size, self.n_agents, self.agent.num_layers, self.agent.hidden_dim,
            )
            return
        initial = self.agent.init_hidden()
        self.hidden_states = initial.unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def forward(self, batch, t):
        batch_size = batch["obs"].shape[0]
        inputs = [batch["obs"][:, t]]
        if self.obs_last_action:
            if t == 0 and not self.shared:
                previous = torch.zeros_like(batch["actions_onehot"][:, t])
                inputs.append(previous)
            elif t == 0:
                try:
                    previous = batch["initial_actions_onehot"]
                except (KeyError, ValueError):
                    previous = torch.zeros_like(batch["actions_onehot"][:, t])
                inputs.append(previous)
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.obs_agent_id:
            identity = torch.eye(self.n_agents, device=batch["obs"].device)
            inputs.append(identity.unsqueeze(0).expand(batch_size, -1, -1))
        inputs = torch.cat(
            [value.reshape(batch_size * self.n_agents, -1) for value in inputs], dim=1,
        )
        if self.multilayer:
            hidden = self.hidden_states.reshape(
                batch_size * self.n_agents, self.agent.num_layers, self.agent.hidden_dim,
            )
            values, hidden = self.agent(inputs, hidden)
            self.hidden_states = hidden.reshape(
                batch_size, self.n_agents, self.agent.num_layers, self.agent.hidden_dim,
            )
        else:
            values, self.hidden_states = self.agent(inputs, self.hidden_states)
        return values.reshape(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other):
        self.agent.load_state_dict(other.agent.state_dict())


@learner_plugin(QmixConfig)
class QMixerLearner(Learner[DefaultEpisodeFields]):
    """QMIX with a monotonic mixer, recurrent utilities and one-step TD."""

    supported_action_spaces = ("discrete",)

    def __init__(self, algo_type, config, agent, env, device, net_source="blanco",
                 model_file_path=None, caller=None, uploader=None, **kwargs):
        supported = AlogsMap.compatible_agent_types(AlogsMap.qmix)
        if agent.agent_type not in supported:
            raise ValueError(f"QMIX supports agent types {supported}, got {agent.agent_type}")
        self.update_cnt = 0
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_target_update_episode = 0
        self.log_stats_t = -config.learner_log_interval - 1
        super().__init__(algo_type, config, agent, env, device, net_source,
                         model_file_path, caller, uploader, **kwargs)

    @property
    def agent_net(self):
        return self.mac.agent

    @property
    def models(self):
        return {"agent_net": self.agent_net, "mixer_net": self.mixer}

    def build_empty_network(self, learner_config, **kwargs):
        if self.agent.config.agent_output_type != "q":
            raise ValueError("QMIX requires agent_output_type='q'")
        if not learner_config.common_reward:
            raise ValueError("QMIX requires a shared team reward")
        state_dim = learner_config.state_shape if self.agent.config.agent_groups else state_shape(self.env)
        self.agent.config.bind_runtime(
            n_agents=self.agent.n_agents, n_actions=self.agent.dim_info.action_dim,
            state_shape=state_dim, device=self.device,
        )
        learner_config.bind_runtime(
            n_agents=self.agent.n_agents, n_actions=self.agent.dim_info.action_dim,
            state_shape=state_dim, hidden_dim=self.agent.config.hidden_dim,
            obs_dim=self.agent.dim_info.obs_dim, device=self.device,
        )
        cfg = learner_config.algo_config
        if cfg.mixer != "qmix":
            raise ValueError(f"QMIX requires mixer='qmix', got {cfg.mixer!r}")
        self.mac = UtilityController(self.agent)
        self.mixer = QMixer(self.agent.n_agents, state_dim, cfg.mixing_embed_dim,
                            cfg.hypernet_layers, cfg.hypernet_embed)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params = list(self.mac.parameters()) + list(self.mixer.parameters())
        self.optimiser = torch.optim.Adam(self.params, lr=cfg.lr)
        self.target_mac = copy.deepcopy(self.mac)
        self.mac.agent.to(self.device)
        self.target_mac.agent.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)
        if learner_config.standardise_rewards:
            self.rew_ms = RunningStatistics((1,), self.device)
        if learner_config.standardise_returns:
            self.ret_ms = RunningStatistics((self.agent.n_agents,), self.device)
        return self.agent_net, self.mixer

    def build_network_from_checkpoint(self, checkpoint, learner_config, **kwargs):
        self.build_empty_network(learner_config, **kwargs)
        dimensions = {"n_agents": self.agent.n_agents, "state_shape": learner_config.state_shape}
        states = {}
        for name, network in self.models.items():
            if name not in checkpoint:
                raise ModelCheckpointCompatibilityException(
                    self.name, name, f"missing network; expected dimensions={dimensions}",
                )
            states[name] = validated_checkpoint_state_dict(
                network, checkpoint[name], agent_name=self.name, model_name=name,
                expected_dimensions=dimensions,
            )
        for name, network in self.models.items():
            network.load_state_dict(states[name])
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        return self.agent_net, self.mixer

    def _offline_net_weights(self):
        return {"mixer_net": self.mixer.state_dict()}

    def snapshot_state(self):
        state = {
            "agent_net": self.agent_net.state_dict(),
            "target_agent_net": self.target_mac.agent.state_dict(),
            "mixer_net": self.mixer.state_dict(),
            "target_mixer_net": self.target_mixer.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "update_cnt": self.update_cnt,
            "training_steps": self.training_steps,
            "last_target_update_step": self.last_target_update_step,
            "last_target_update_episode": self.last_target_update_episode,
            "log_stats_t": self.log_stats_t,
        }
        for name in ("rew_ms", "ret_ms"):
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def restore_state(self, state):
        self.agent_net.load_state_dict(state["agent_net"])
        self.target_mac.agent.load_state_dict(state["target_agent_net"])
        self.mixer.load_state_dict(state["mixer_net"])
        self.target_mixer.load_state_dict(state["target_mixer_net"])
        self.optimiser.load_state_dict(state["optimiser"])
        for name in ("update_cnt", "training_steps", "last_target_update_step",
                     "last_target_update_episode", "log_stats_t"):
            setattr(self, name, state[name])
        for name in ("rew_ms", "ret_ms"):
            if name in state:
                setattr(self, name, state[name])

    def dynamic_modify_config(self, updates):
        super().dynamic_modify_config(updates)
        for group in self.optimiser.param_groups:
            group["lr"] = self.config.algo_config.lr

    def _make_batch(self, episodes, max_seq_len):
        return EpisodeTransition.to_batch(
            agent_ids=self.agent.agent_list, episode_trans=episodes,
            expected_n_agents=self.agent.n_agents, max_seq_len=max_seq_len,
            action_dim=self.agent.dim_info.action_dim, device=self.device,
        )

    def _schema_batch(self, fields, max_seq_len):
        """Translate the shared layout into QMIX's internal recurrent inputs."""
        batch_size, steps, n_agents, _ = fields["actions"].shape
        width = max(steps, max_seq_len)
        actions_count = self.agent.dim_info.action_dim
        frames = (batch_size, width + 1)
        result = EpisodeBatch({
            "obs": torch.zeros(*frames, n_agents, fields["obs"].shape[-1], device=self.device),
            "state": torch.zeros(*frames, fields["state"].shape[-1], device=self.device),
            "actions": torch.zeros(*frames, n_agents, 1, dtype=torch.long, device=self.device),
            "actions_onehot": torch.zeros(*frames, n_agents, actions_count, device=self.device),
            "avail_actions": torch.zeros(*frames, n_agents, actions_count, device=self.device),
            "reward": torch.zeros(*frames, 1, device=self.device),
            "terminated": torch.zeros(*frames, 1, device=self.device),
            "filled": torch.zeros(*frames, 1, dtype=torch.long, device=self.device),
            "active_agents": torch.zeros(*frames, n_agents, 1, device=self.device),
        })
        prefixes = fields.get("burn_in", [None] * batch_size)
        if any(prefix is not None for prefix in prefixes):
            result.data["initial_actions_onehot"] = torch.zeros(
                batch_size, n_agents, actions_count, device=self.device,
            )
        result["obs"][:, :steps + 1].copy_(fields["obs"])
        result["state"][:, :steps + 1].copy_(fields["state"])
        result["actions"][:, :steps].copy_(fields["actions"])
        result["reward"][:, :steps, 0].copy_(fields["team_rewards"])
        result["terminated"][:, :steps, 0].copy_(fields["team_terminated"])
        result["filled"][:, :steps, 0].copy_(fields["valid"])
        result["active_agents"][:, :steps + 1, :, 0].copy_(fields["active"])
        result["avail_actions"].fill_(1)
        result["avail_actions"][:, :steps + 1].copy_(fields["action_masks"])
        onehot = torch.nn.functional.one_hot(fields["actions"].squeeze(-1), actions_count)
        result["actions_onehot"][:, :steps].copy_(onehot * fields["valid"].unsqueeze(-1).unsqueeze(-1))
        if "initial_actions_onehot" in result.data:
            for row, prefix in enumerate(prefixes):
                if prefix is None:
                    continue
                length = int(prefix["valid"][0].sum().item())
                final = prefix["actions"][0, length - 1, :, 0]
                active = prefix["active"][0, length - 1]
                result["initial_actions_onehot"][row].copy_(
                    torch.nn.functional.one_hot(final, actions_count) * active.unsqueeze(-1)
                )
        return result

    @torch.no_grad()
    def _schema_initial_states(self, fields):
        prefixes = fields["burn_in"]
        histories = [(index, prefix) for index, prefix in enumerate(prefixes) if prefix is not None]
        for index, prefix in enumerate(prefixes):
            if prefix is None and not fields["starts_with_episode"][index]:
                raise ReplayBatchShapeException(self.name, "中途开始的 RNN 轨迹缺少同组局首 burn-in 历史。")
        if not histories:
            return None, None
        lengths = [int(prefix["valid"][0].sum().item()) for _, prefix in histories]
        width = max(lengths)
        batches = [self._schema_batch(prefix, width) for _, prefix in histories]
        combined = EpisodeBatch({
            name: torch.cat([item.data[name] for item in batches], dim=0)
            for name in batches[0].data
        })
        states = []
        for controller in (self.mac, self.target_mac):
            controller.init_hidden(len(batches))
            restored = controller.hidden_states.clone()
            for time in range(width):
                controller.forward(combined, time)
                hidden = controller.hidden_states.reshape(
                    len(batches), self.agent.n_agents, *restored.shape[2:],
                )
                for row, length in enumerate(lengths):
                    if time + 1 == length:
                        restored[row].copy_(hidden[row])
            controller.init_hidden(fields["actions"].shape[0])
            initial = controller.hidden_states.detach().clone()
            initial[[index for index, _ in histories]] = restored.detach()
            states.append(initial)
        return tuple(states)

    @torch.no_grad()
    def _slice_initial_states(self, episodes):
        histories, indices = [], []
        for index, episode in enumerate(episodes):
            prefix = episode.burn_in
            if prefix is None:
                if not episode.starts_with_episode:
                    raise ReplayBatchShapeException(self.name, "中途开始的 RNN 轨迹缺少同组局首 burn-in 历史。")
                continue
            if (not isinstance(prefix, EpisodeTransition) or prefix.seq_len <= 0
                    or tuple(prefix.agent_ids) != tuple(episode.agent_ids)
                    or prefix.episode_num != episode.episode_num
                    or not prefix.starts_with_episode
                    or prefix.burn_in is not None or episode.seq_len <= 0):
                raise ReplayBatchShapeException(self.name, "burn-in 必须是同组、同局、从 reset 开始的历史。")
            if (not np.array_equal(prefix.next_states[-1], episode.states[0])
                    or any(not np.array_equal(prefix.next_obs[aid][-1], episode.obs[aid][0])
                           for aid in episode.agent_ids)):
                raise ReplayBatchShapeException(self.name, "burn-in 末帧与切片首帧不连续。")
            histories.append(prefix)
            indices.append(index)
        if not histories:
            return None, None
        batch = self._make_batch(histories, max(history.seq_len for history in histories))
        results = []
        for controller in (self.mac, self.target_mac):
            controller.init_hidden(batch["obs"].shape[0])
            restored = controller.hidden_states.clone()
            for time in range(max(history.seq_len for history in histories)):
                controller.forward(batch, time)
                hidden = controller.hidden_states.reshape(
                    batch["obs"].shape[0], self.agent.n_agents, *restored.shape[2:],
                )
                for row, history in enumerate(histories):
                    if time + 1 == history.seq_len:
                        restored[row].copy_(hidden[row])
            controller.init_hidden(len(episodes))
            initial = controller.hidden_states.detach().clone()
            initial[indices] = restored.detach()
            results.append(initial)
        return tuple(results)

    def update(self, episodes: EpisodeBatch[DefaultEpisodeFields]):
        if isinstance(episodes, EpisodeBatch):
            initial, target_initial = self._schema_initial_states(episodes.data)
            batch = self._schema_batch(episodes.data, self.env.config.max_step_per_episode)
        else:
            initial, target_initial = self._slice_initial_states(episodes)
            batch = self._make_batch(episodes, self.env.config.max_step_per_episode)
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        available = batch["avail_actions"]
        try:
            active = batch["active_agents"][:, :-1].float().squeeze(-1)
            next_active = batch["active_agents"][:, 1:].float().squeeze(-1)
        except KeyError:
            active = torch.ones_like(actions.squeeze(-1), dtype=torch.float32)
            next_active = torch.ones_like(active)

        if self.config.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)
        rewards = rewards.expand(-1, -1, self.agent.n_agents)

        self.mac.init_hidden(batch["obs"].shape[0])
        if initial is not None:
            self.mac.hidden_states = initial.detach().clone()
        values = torch.stack([self.mac.forward(batch, t) for t in range(batch["obs"].shape[1])], dim=1)
        selected = torch.gather(values[:, :-1], dim=3, index=actions).squeeze(3) * active

        self.target_mac.init_hidden(batch["obs"].shape[0])
        if target_initial is not None:
            self.target_mac.hidden_states = target_initial.detach().clone()
        targets = [self.target_mac.forward(batch, t) for t in range(batch["obs"].shape[1])]
        targets = torch.stack(targets[1:], dim=1)
        targets[available[:, 1:] == 0] = -9999999
        if self.config.algo_config.double_q:
            online = values.clone().detach()
            online[available == 0] = -9999999
            best = online[:, 1:].max(dim=3, keepdim=True)[1]
            target_values = torch.gather(targets, 3, best).squeeze(3)
        else:
            target_values = targets.max(dim=3)[0]
        target_values = target_values * next_active
        chosen_total = self.mixer(selected, batch["state"][:, :-1])
        target_total = self.target_mixer(target_values, batch["state"][:, 1:])
        if self.config.standardise_returns:
            target_total = target_total * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean
        expected = rewards + self.config.gamma * (1 - terminated) * target_total.detach()
        if self.config.standardise_returns:
            self.ret_ms.update(expected)
            expected = (expected - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)
        error = chosen_total - expected.detach()
        mask = mask.expand_as(error)
        masked_error = error * mask
        loss = masked_error.square().sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.params, self.config.grad_norm_clip)
        self.optimiser.step()
        self.training_steps += 1
        update_interval = self.config.algo_config.target_update_interval_or_tau
        if update_interval > 1 and (self.training_steps - self.last_target_update_step) / update_interval >= 1:
            self.target_mac.load_state(self.mac)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.last_target_update_step = self.training_steps
        elif update_interval <= 1:
            for target, source in zip(self.target_mac.parameters(), self.mac.parameters()):
                target.data.copy_(target.data * (1.0 - update_interval) + source.data * update_interval)
            for target, source in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target.data.copy_(target.data * (1.0 - update_interval) + source.data * update_interval)

        if self.update_cnt - self.log_stats_t >= self.config.learner_log_interval:
            count = mask.sum().item()
            LOG.log_stat("loss", loss.item(), self.update_cnt)
            LOG.log_stat("grad_norm", grad_norm.item(), self.update_cnt)
            LOG.log_stat("td_error_abs", masked_error.abs().sum().item() / count, self.update_cnt)
            LOG.log_stat("q_taken_mean", (chosen_total * mask).sum().item() / (count * self.agent.n_agents),
                         self.update_cnt)
            LOG.log_stat("target_mean", (expected * mask).sum().item() / (count * self.agent.n_agents),
                         self.update_cnt)
            self.log_stats_t = self.update_cnt
        self.update_cnt += 1
        return LOG.pop_stats()
