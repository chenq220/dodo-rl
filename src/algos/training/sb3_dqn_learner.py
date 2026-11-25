from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from stable_baselines3 import DQN
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import FloatSchedule, polyak_update
from stable_baselines3.dqn.policies import DQNPolicy

from algos.agents import Agent
from algos.learners import Learner
from algos.map import AlogsMap
from config import DqnConfig, LearnerConfig
from common.plugins import learner_plugin
from envs.base import BaseEnvironment
from sample.model import EpisodeBatch, SABatchTensorData
from algos.learners.batch_schema import SINGLE_AGENT_SCHEMA, SingleAgentFields
from sample.sample_transfer import SAEpisodeTransition
from utils.checkpoint import extract_state_dict

if TYPE_CHECKING:
    from common.types import FilePath, ModelSource, NetWeight


class _SpaceEnv(gym.Env):
    """只向 SB3 DQN 提供观测/动作空间；框架采样流程不会调用该环境的 step/reset。"""

    metadata = {"render_modes": []}

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


# IDEA: SB3的DQN算法太基础，不支持Double/Dueling/Rainbow扩展，不支持动作掩码，后续看需求重写一个功能全面的

# SB3DQNLearner：单智能体 DQN 学习器，基于 stable_baselines3.DQN。
# 框架 ReplayBuffer 仍按 episode 存储样本；update 时只把尚未写入过的 transition
# 同步到 SB3 内部 replay buffer，避免同一条历史样本被采样策略反复放大。
# SB3 的 target network 原本由 learn() 的环境步进回调同步，这里因为不调用 learn()，
# 所以在 update 后显式按配置同步目标网络。
@learner_plugin(DqnConfig)
class SB3DQNLearner(Learner[SingleAgentFields]):
    """DODORL Learner 到 stable_baselines3 DQN 的单智能体适配层。"""
    batch_schema = SINGLE_AGENT_SCHEMA

    def __init__(
            self,
            algo_type: str,
            config: LearnerConfig,
            agent: Agent,
            env: BaseEnvironment,
            device: Optional[Union[torch.device, str]],
            net_source: ModelSource = "blanco",
            model_file_path: Dict[str, FilePath] = None,
            caller: Callable[[bool], Dict[str, NetWeight]] = None,
            uploader: Callable[[Dict[str, nn.Module]], None] = None,
            **kwargs,
    ):
        supported_agent_types = AlogsMap.compatible_agent_types(AlogsMap.dqn)
        assert agent.agent_type in supported_agent_types, (
            f"SB3DQNLearner supports agent types {supported_agent_types}, "
            f"given={agent.agent_type}, learner={config.name}"
        )
        self._dqn: Optional[DQN] = None
        self.update_cnt = 0
        self._seen_replay_transition_keys: set[str] = set()
        self._seen_replay_transition_order: deque[str] = deque()
        super().__init__(
            algo_type=algo_type,
            config=config,
            agent=agent,
            env=env,
            device=device,
            net_source=net_source,
            model_file_path=model_file_path,
            caller=caller,
            uploader=uploader,
            **kwargs,
        )

    @property
    def dqn(self) -> DQN:
        assert self._dqn is not None, "SB3DQNLearner.dqn must be initialized before access."
        return self._dqn

    @property
    def agent_net(self) -> nn.Module:
        return self.dqn.policy

    @property
    def models(self) -> Dict[str, nn.Module]:
        return {"agent_net": self.dqn.policy}

    def snapshot_state(self) -> Dict[str, Any]:
        """
        保存 DQN 全部训练状态：policy state_dict（含 q_net 与 q_net_target）、优化器 state_dict、
        update 计数器、去重集合与顺序队列、SB3 内部更新步数/时间步、以及 SB3 内部回放池。
        DQN 的 SB3 回放池跨 update 累积样本，任务重启时必须连同去重集合一起恢复，否则会重复灌入旧样本。
        """
        return {
            "policy_state": self.dqn.policy.state_dict(),
            "optimizer_state": self.dqn.policy.optimizer.state_dict(),
            "update_cnt": self.update_cnt,
            "_seen_replay_transition_keys": self._seen_replay_transition_keys,
            "_seen_replay_transition_order": self._seen_replay_transition_order,
            "dqn_n_updates": self.dqn._n_updates,
            "dqn_num_timesteps": self.dqn.num_timesteps,
            "dqn_replay_buffer": self.dqn.replay_buffer,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.dqn.policy.load_state_dict(state["policy_state"])
        self.dqn.policy.optimizer.load_state_dict(state["optimizer_state"])
        self.update_cnt = state["update_cnt"]
        self._seen_replay_transition_keys = state["_seen_replay_transition_keys"]
        self._seen_replay_transition_order = state["_seen_replay_transition_order"]
        self.dqn._n_updates = state["dqn_n_updates"]
        self.dqn.num_timesteps = state["dqn_num_timesteps"]
        self.dqn.replay_buffer = state["dqn_replay_buffer"]

    def build_empty_network(self, learner_config: LearnerConfig, **kwargs) -> Tuple[nn.Module, ...]:
        self._validate_agent(learner_config)
        self._dqn = self._build_dqn(learner_config)
        self._assert_policy()
        return (self.dqn.policy,)

    def build_network_from_checkpoint(
            self,
            checkpoint: Dict[str, NetWeight],
            learner_config: LearnerConfig,
            **kwargs,
    ) -> Tuple[nn.Module, ...]:
        self.build_empty_network(learner_config, **kwargs)
        self.dqn.policy.load_state_dict(extract_state_dict(checkpoint["agent_net"]), strict=False)
        self.dqn.q_net_target.load_state_dict(self.dqn.q_net.state_dict())
        self._assert_policy()
        return (self.dqn.policy,)

    def _offline_net_weights(self) -> Dict[str, Dict]:
        # DQN target network 是训练辅助网络；执行侧只需要在线 q_net 所在的 policy。
        return {}

    def update(self, batch_data: EpisodeBatch[SingleAgentFields]) -> Dict[str, Any]:
        batch = (batch_data if isinstance(batch_data, EpisodeBatch)
                 else SAEpisodeTransition.to_batch_tensor(episode_trans=batch_data, device=self.device))
        self._assert_batch_finite(batch)
        if self._has_dynamic_action_mask(batch):
            return self._masked_dqn_update(batch)

        if isinstance(batch_data, EpisodeBatch):
            inserted, duplicate_skipped = self._append_batch_to_replay_buffer(batch)
        else:
            inserted, duplicate_skipped = self._append_to_replay_buffer(batch_data)

        algo_config = self.config.algo_config
        assert isinstance(algo_config, DqnConfig)
        replay_size = self.dqn.replay_buffer.size()
        if replay_size < max(algo_config.learning_starts, algo_config.train_batch_size):
            return {
                "skip_update": True,
                "skip_reason": "dqn_replay_not_ready",
                "inserted": inserted,
                "duplicate_skipped": duplicate_skipped,
                "replay_size": replay_size,
                "seen_transitions": len(self._seen_replay_transition_keys),
            }

        self.dqn.train(
            gradient_steps=algo_config.gradient_steps,
            batch_size=algo_config.train_batch_size,
        )
        self.update_cnt += 1
        self._maybe_update_target_network(algo_config)
        return self._collect_stats(inserted=inserted, duplicate_skipped=duplicate_skipped)

    def dynamic_modify_config(self, updates: Dict[str, Any]) -> None:
        """按学习令牌同步 DQN 超参；policy/hidden_dim/buffer_size 等结构型参数不允许运行期变更。"""
        if not updates:
            return
        current = self.config.algo_config
        assert isinstance(current, DqnConfig), (
            f"SB3DQNLearner.config.algo_config must be DqnConfig, given={type(current).__name__}"
        )
        if "policy" in updates and updates["policy"] != current.policy:
            raise ValueError("DQN policy cannot be modified without rebuilding the network.")
        if "hidden_dim" in updates and updates["hidden_dim"] != current.hidden_dim:
            raise ValueError("DQN hidden_dim cannot be modified without rebuilding the network.")
        if "buffer_size" in updates and updates["buffer_size"] != current.buffer_size:
            raise ValueError("DQN buffer_size cannot be modified without rebuilding the network.")
        super().dynamic_modify_config(updates)
        algo_configs = self.config.algo_config
        self.dqn.learning_rate = algo_configs.lr
        self.dqn.lr_schedule = FloatSchedule(algo_configs.lr)
        self.dqn.batch_size = algo_configs.train_batch_size
        self.dqn.tau = algo_configs.tau
        self.dqn.gamma = self.config.gamma
        self.dqn.gradient_steps = algo_configs.gradient_steps
        self.dqn.target_update_interval = algo_configs.target_update_interval
        self.dqn.max_grad_norm = algo_configs.max_grad_norm
        for param_group in self.dqn.policy.optimizer.param_groups:
            param_group["lr"] = algo_configs.lr


    def _build_dqn(self, learner_config: LearnerConfig) -> DQN:
        algo_config = learner_config.algo_config
        assert isinstance(algo_config, DqnConfig), (
            f"SB3DQNLearner requires DqnConfig, given={type(algo_config).__name__}"
        )
        space_env = _SpaceEnv(self.agent.obs_dim, self.agent.action_dim)
        dqn = DQN(
            policy=algo_config.policy,
            env=space_env,
            learning_rate=algo_config.lr,
            buffer_size=algo_config.buffer_size,
            learning_starts=algo_config.learning_starts,
            batch_size=algo_config.train_batch_size,
            tau=algo_config.tau,
            gamma=learner_config.gamma,
            train_freq=algo_config.train_freq,
            gradient_steps=algo_config.gradient_steps,
            target_update_interval=algo_config.target_update_interval,
            exploration_fraction=algo_config.exploration_fraction,
            exploration_initial_eps=algo_config.exploration_initial_eps,
            exploration_final_eps=algo_config.exploration_final_eps,
            max_grad_norm=algo_config.max_grad_norm,
            policy_kwargs={"net_arch": [algo_config.hidden_dim, algo_config.hidden_dim]},
            device=str(self.device),
            verbose=0,
        )
        dqn.set_logger(Logger(folder=None, output_formats=[]))
        return dqn

    def _mark_transition_seen(self, key: str) -> None:
        """
        记录已写入SB3 replay的transition指纹，并按SB3 replay容量限制索引内存。

        DQN的内部replay buffer是环形缓存；指纹索引也保持同量级容量，避免长时间训练时
        seen集合无限增长。被淘汰的旧指纹如果再次从外层ReplayBuffer采到，允许重新写入，
        语义上等价于SB3 replay中旧transition被新transition覆盖后的再次进入。
        """
        algo_config = self.config.algo_config
        assert isinstance(algo_config, DqnConfig)
        self._seen_replay_transition_keys.add(key)
        self._seen_replay_transition_order.append(key)
        while len(self._seen_replay_transition_order) > algo_config.buffer_size:
            stale_key = self._seen_replay_transition_order.popleft()
            self._seen_replay_transition_keys.discard(stale_key)

    def _masked_dqn_update(self, batch: SABatchTensorData) -> Dict[str, Any]:
        """
        动态动作掩码场景下的 DQN 更新。

        SB3基础DQN的ReplayBuffer不保存下一状态动作掩码，无法在target里屏蔽非法动作。
        这里直接使用框架ReplayBuffer抽出的batch执行TD更新，target计算时用
        batch.avail_actions屏蔽next_obs上的非法动作。
        """
        algo_config = self.config.algo_config
        assert isinstance(algo_config, DqnConfig)

        losses: List[float] = []
        grad_norms: List[float] = []
        q_taken_means: List[float] = []
        target_means: List[float] = []
        td_error_abs: List[float] = []
        masked_action_ratios: List[float] = []

        for _ in range(max(algo_config.gradient_steps, 1)):
            indices = self._sample_batch_indices(batch.obs.shape[0], algo_config.train_batch_size)
            train_batch = self._index_batch(batch, indices)
            loss, stats = self._masked_dqn_train_step(train_batch, algo_config)
            losses.append(float(loss))
            grad_norms.append(float(stats["grad_norm"]))
            q_taken_means.append(float(stats["q_taken_mean"]))
            target_means.append(float(stats["target_mean"]))
            td_error_abs.append(float(stats["td_error_abs"]))
            masked_action_ratios.append(float(stats["masked_action_ratio"]))

        self.update_cnt += 1
        self._maybe_update_target_network(algo_config)
        return {
            "train/loss": float(np.mean(losses)),
            "train/grad_norm": float(np.mean(grad_norms)),
            "train/q_taken_mean": float(np.mean(q_taken_means)),
            "train/target_mean": float(np.mean(target_means)),
            "train/td_error_abs": float(np.mean(td_error_abs)),
            "train/masked_action_ratio": float(np.mean(masked_action_ratios)),
            "train/gradient_steps": algo_config.gradient_steps,
            "update_cnt": self.update_cnt,
            "inserted": 0,
            "duplicate_skipped": 0,
            "replay_size": 0,
            "seen_transitions": len(self._seen_replay_transition_keys),
            "masked_dqn": True,
        }

    def _masked_dqn_train_step(
            self,
            batch: SABatchTensorData,
            algo_config: DqnConfig,
    ) -> Tuple[float, Dict[str, float]]:
        """执行一次支持next_avail_actions的DQN TD更新。"""
        assert batch.avail_actions is not None
        obs = batch.obs.to(self.device)
        next_obs = batch.next_obs.to(self.device)
        actions = batch.actions.to(self.device).long().reshape(-1, 1)
        rewards = batch.rewards.to(self.device).reshape(-1, 1)
        terminated = batch.terminated.to(self.device).reshape(-1, 1)
        next_avail_actions = batch.avail_actions.to(self.device).bool()

        q_values = self.dqn.q_net(obs)
        q_taken = q_values.gather(dim=1, index=actions)

        with torch.no_grad():
            target_q_values = self.dqn.q_net_target(next_obs)
            if next_avail_actions.shape != target_q_values.shape:
                raise ValueError(
                    f"next action mask shape mismatch, mask={tuple(next_avail_actions.shape)}, "
                    f"q_values={tuple(target_q_values.shape)}"
                )
            if not next_avail_actions.any(dim=1).all():
                raise ValueError("SB3DQNLearner masked DQN received next state with no available action.")
            target_q_values = target_q_values.masked_fill(~next_avail_actions, torch.finfo(target_q_values.dtype).min)
            next_q = target_q_values.max(dim=1, keepdim=True).values
            target = rewards + self.config.gamma * (1.0 - terminated) * next_q

        loss = F.smooth_l1_loss(q_taken, target)
        self.dqn.policy.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.dqn.policy.parameters(), algo_config.max_grad_norm)
        self.dqn.policy.optimizer.step()

        td_error = (q_taken.detach() - target).abs()
        masked_action_ratio = 1.0 - next_avail_actions.float().mean().item()
        return float(loss.detach().cpu().item()), {
            "grad_norm": float(grad_norm),
            "q_taken_mean": float(q_taken.detach().mean().cpu().item()),
            "target_mean": float(target.detach().mean().cpu().item()),
            "td_error_abs": float(td_error.mean().cpu().item()),
            "masked_action_ratio": float(masked_action_ratio),
        }

    @staticmethod
    def _sample_batch_indices(num_transitions: int, batch_size: int) -> torch.Tensor:
        """从当前框架batch中抽取一次DQN训练mini-batch。"""
        if num_transitions <= 0:
            raise ValueError("SB3DQNLearner received empty DQN batch.")
        sample_size = min(num_transitions, batch_size)
        if num_transitions <= sample_size:
            return torch.arange(num_transitions, dtype=torch.long)
        return torch.randperm(num_transitions, dtype=torch.long)[:sample_size]

    @staticmethod
    def _index_batch(batch: SABatchTensorData, indices: torch.Tensor) -> SABatchTensorData:
        """按transition索引切出SABatchTensorData子批次。"""
        device_indices = indices.to(batch.obs.device)
        return SABatchTensorData(dict(
            obs=batch.obs[device_indices],
            actions=batch.actions[device_indices],
            last_actions=batch.last_actions[device_indices],
            rewards=batch.rewards[device_indices],
            next_obs=batch.next_obs[device_indices],
            terminated=batch.terminated[device_indices],
            truncated=batch.truncated[device_indices],
            episode_starts=batch.episode_starts[device_indices],
            avail_actions=None if batch.avail_actions is None else batch.avail_actions[device_indices],
            old_log_probs=None if batch.old_log_probs is None else batch.old_log_probs[device_indices],
            old_values=None if batch.old_values is None else batch.old_values[device_indices],
            returns=None if batch.returns is None else batch.returns[device_indices],
            advantages=None if batch.advantages is None else batch.advantages[device_indices],
        ))

    def _append_to_replay_buffer(self, batch_data: List[SAEpisodeTransition]) -> Tuple[int, int]:
        """
        把框架单智能体样本同步到 SB3 replay buffer。

        当前框架的学习入口仍会按采样策略反复给 DQN 传历史 episode。DQN 自己已经有
        replay buffer，如果每次 update 都把抽到的历史样本再次写入，会把外层采样策略
        的偏置叠加到 SB3 replay 中。因此这里按 transition 内容生成指纹，只允许同一条
        环境转移写入一次。
        """
        inserted = 0
        duplicate_skipped = 0
        for episode in batch_data:
            for step in range(episode.seq_len):
                key = self._transition_key(episode, step)
                if key in self._seen_replay_transition_keys:
                    duplicate_skipped += 1
                    continue
                self._mark_transition_seen(key)
                self._add_transition_to_replay_buffer(episode, step)
                inserted += 1
        return inserted, duplicate_skipped

    def _append_batch_to_replay_buffer(self, batch: EpisodeBatch) -> Tuple[int, int]:
        inserted = duplicate_skipped = 0
        for index, key in enumerate(batch.metadata["transition_keys"]):
            if key in self._seen_replay_transition_keys:
                duplicate_skipped += 1
                continue
            self._mark_transition_seen(key)
            terminated = bool(batch.terminated[index].item())
            truncated = bool(batch.truncated[index].item())
            self.dqn.replay_buffer.add(
                obs=batch.obs[index].detach().cpu().numpy(),
                next_obs=batch.next_obs[index].detach().cpu().numpy(),
                action=batch.actions[index].detach().cpu().numpy(),
                reward=np.asarray([float(batch.rewards[index].item())], dtype=np.float32),
                done=np.asarray([terminated or truncated], dtype=bool),
                infos=[{"TimeLimit.truncated": truncated and not terminated}],
            )
            inserted += 1
        return inserted, duplicate_skipped

    def _add_transition_to_replay_buffer(self, episode: SAEpisodeTransition, step: int) -> None:
        """将单条transition按SB3 ReplayBuffer.add需要的格式写入。"""
        obs = np.asarray(episode.obs[step], dtype=np.float32).flatten()
        next_obs = np.asarray(episode.next_obs[step], dtype=np.float32).flatten()
        action = np.asarray([int(episode.actions[step])], dtype=np.int64)
        reward = np.asarray([float(episode.rewards[step])], dtype=np.float32)
        terminated = bool(episode.terminations[step])
        truncated = bool(episode.truncations[step])
        done = np.asarray([terminated or truncated], dtype=bool)
        info = {"TimeLimit.truncated": bool(truncated and not terminated)}
        self.dqn.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=[info],
        )

    @staticmethod
    def _transition_key(episode: SAEpisodeTransition, step: int) -> str:
        """
        生成单条transition指纹。

        按步采样会把一个episode切成多个chunk，chunk内step会重新从0开始，所以不能只用
        episode_num + step。这里把episode_num和转移内容一起写入，既避免同一条样本重复写入，
        也避免不同episode中偶然相同的观测被误判为重复。
        """
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(getattr(episode, "episode_num", 0)).encode("utf-8"))
        for value in (
                episode.obs[step],
                int(episode.actions[step]),
                float(episode.rewards[step]),
                episode.next_obs[step],
                bool(episode.terminations[step]),
                bool(episode.truncations[step]),
        ):
            SB3DQNLearner._update_digest_value(digest, value)
        return digest.hexdigest()

    @staticmethod
    def _update_digest_value(digest: "hashlib._Hash", value: Any) -> None:
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            digest.update(str(arr.shape).encode("utf-8"))
            digest.update(str(arr.dtype).encode("utf-8"))
            digest.update(arr.tobytes())
            return
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            SB3DQNLearner._update_digest_value(digest, value.detach().cpu().numpy())
            return
        if isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode("utf-8"))
            for item in value:
                SB3DQNLearner._update_digest_value(digest, item)
            return
        digest.update(repr(value).encode("utf-8"))

    def _maybe_update_target_network(self, algo_config: DqnConfig) -> None:
        if self.update_cnt % max(algo_config.target_update_interval, 1) != 0:
            return
        polyak_update(self.dqn.q_net.parameters(), self.dqn.q_net_target.parameters(), algo_config.tau)
        polyak_update(self.dqn.batch_norm_stats, self.dqn.batch_norm_stats_target, 1.0)

    def _collect_stats(self, inserted: int, duplicate_skipped: int) -> Dict[str, Any]:
        from scheduler.model import normalize_sb3_learn_metrics
        stats = normalize_sb3_learn_metrics(
            {key: value for key, value in self.dqn.logger.name_to_value.items()}
        )
        stats["update_cnt"] = self.update_cnt
        stats["inserted"] = inserted
        stats["duplicate_skipped"] = duplicate_skipped
        stats["replay_size"] = self.dqn.replay_buffer.size()
        stats["seen_transitions"] = len(self._seen_replay_transition_keys)
        return stats

    @staticmethod
    def _assert_batch_finite(batch: SABatchTensorData) -> None:
        required_tensors = {
            "obs": batch.obs,
            "actions": batch.actions,
            "rewards": batch.rewards,
            "next_obs": batch.next_obs,
            "terminated": batch.terminated,
            "truncated": batch.truncated,
        }
        for name, tensor in required_tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"SB3DQNLearner received non-finite batch tensor: {name}")

    @staticmethod
    def _has_dynamic_action_mask(batch: SABatchTensorData) -> bool:
        """
        判断batch是否包含动态动作掩码。

        单智能体样本中的avail_actions来自next_obs对应的动作表状态；若存在0，说明TD
        target必须屏蔽下一状态非法动作，走自定义masked DQN更新。
        """
        if batch.avail_actions is None:
            return False
        return bool(torch.any(batch.avail_actions <= 0).item())

    def _validate_agent(self, learner_config: LearnerConfig) -> None:
        if not isinstance(learner_config.algo_config, DqnConfig):
            raise TypeError(
                f"SB3DQNLearner requires DqnConfig, "
                f"given={type(learner_config.algo_config).__name__}"
            )
        if self.agent.config.agent_output_type != "q":
            raise ValueError("SB3DQNLearner requires agent_output_type='q'")
        if self.agent.n_agents != 1:
            raise ValueError(
                f"SB3DQNLearner is single-agent only, expected agent.n_agents == 1, "
                f"got {self.agent.n_agents} (agent_list={self.agent.agent_list})"
            )

    def _assert_policy(self) -> None:
        assert isinstance(self.dqn, DQN), f"self.dqn must be DQN, given={type(self._dqn).__name__}"
        assert isinstance(self.dqn.policy, DQNPolicy), (
            f"self.dqn.policy must be DQNPolicy, given={type(self.dqn.policy).__name__}"
        )
