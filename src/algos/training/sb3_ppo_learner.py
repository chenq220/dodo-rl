from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import FloatSchedule

from algos.agents import Agent
from algos.map import AlogsMap
from algos.learners.base import Learner, LOG
from config import LearnerConfig, PpoConfig
from common.plugins import learner_plugin
from envs.base import BaseEnvironment
from sample.model import EpisodeBatch, SABatchTensorData
from algos.learners.batch_schema import SINGLE_AGENT_SCHEMA, SingleAgentFields
from sample.sample_transfer import SAEpisodeTransition
from utils.checkpoint import extract_state_dict
from utils.transition_hash import transition_batch_hash

if TYPE_CHECKING:
    from common.types import FilePath, ModelSource, NetWeight


class _SpaceEnv(gym.Env):
    """
    仅承载 observation_space / action_space 的占位环境。

    stable_baselines3.PPO 构造时强制要求一个 env 来确定空间维度并构建策略网络。
    本项目的 PPO 是离线训练(样本来自框架 ReplayBuffer)，永远不会真正 step/reset 这个 env，
    因此这里只提供空间定义和合法的 reset/step 桩函数，避免 SB3 在包装 DummyVecEnv 时报错。
    """

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


# SB3PPOLearner：单智能体 PPO 学习器，基于 stable_baselines3 的 PPO 实现。
# 它把框架采样到的 SAEpisodeTransition 拍平成 SABatchTensorData，再灌进 SB3 的 RolloutBuffer，
# 用当前策略补算 old_log_prob / value 并做 per-episode GAE，最后直接复用 SB3 PPO 的裁剪式更新。
# 对外暴露的 agent_net 是 SB3 ActorCriticPolicy(含 actor 与 critic)，供单智能体 PPO Agent 克隆执行。
@learner_plugin(PpoConfig)
class SB3PPOLearner(Learner[SingleAgentFields]):
    """
    项目 Learner 接口到 stable_baselines3 PPO 的适配层（单智能体）。

    Learner 内部持有一个 SB3 PPO 实例；训练数据来自框架而非 SB3 自带的 rollout 采集，
    所以 update 时会用样本重建 RolloutBuffer 后调用 PPO.train()，不触碰 PPO.learn()。
    """
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
            **kwargs
    ):
        supported_agent_types = AlogsMap.compatible_agent_types(AlogsMap.ppo)
        assert agent.agent_type in supported_agent_types, (
            f"SB3PPOLearner supports agent types {supported_agent_types}, "
            f"given={agent.agent_type}, learner={config.name}"
        )
        self._ppo: Optional[PPO] = None
        self.update_cnt = 0
        self._last_batch_hash: Optional[str] = None
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
    def ppo(self) -> PPO:
        assert self._ppo is not None, "SB3PPOLearner.ppo must be initialized before access."
        return self._ppo

    @property
    def agent_net(self) -> nn.Module:
        return self.ppo.policy

    @property
    def models(self) -> Dict[str, nn.Module]:
        # SB3 ActorCriticPolicy 内部同时包含 actor 与 critic，单个 policy 即为完整可执行网络。
        return {"agent_net": self.ppo.policy}

    def snapshot_state(self) -> Dict[str, Any]:
        """
        保存 PPO 全部训练状态：ActorCriticPolicy state_dict（含 actor/critic）、
        优化器 state_dict、update 计数器、防重复批次哈希、SB3 内部更新步数与时间步。
        """
        return {
            "policy_state": self.ppo.policy.state_dict(),
            "optimizer_state": self.ppo.policy.optimizer.state_dict(),
            "update_cnt": self.update_cnt,
            "_last_batch_hash": self._last_batch_hash,
            "ppo_n_updates": self.ppo._n_updates,
            "ppo_num_timesteps": self.ppo.num_timesteps,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.ppo.policy.load_state_dict(state["policy_state"])
        self.ppo.policy.optimizer.load_state_dict(state["optimizer_state"])
        self.update_cnt = state["update_cnt"]
        self._last_batch_hash = state["_last_batch_hash"]
        self.ppo._n_updates = state["ppo_n_updates"]
        self.ppo.num_timesteps = state["ppo_num_timesteps"]

    def build_empty_network(self, learner_config: LearnerConfig, **kwargs) -> Tuple[nn.Module, ...]:
        self._validate_agent(learner_config)
        self._ppo = self._build_ppo(learner_config)
        self._assert_policy()
        return (self.ppo.policy,)

    def build_network_from_checkpoint(
            self,
            checkpoint: Dict[str, NetWeight],
            learner_config: LearnerConfig,
            **kwargs
    ) -> Tuple[nn.Module, ...]:
        self.build_empty_network(learner_config, **kwargs)
        self.ppo.policy.load_state_dict(extract_state_dict(checkpoint["agent_net"]), strict=False)
        self._assert_policy()
        return (self.ppo.policy,)

    def _offline_net_weights(self) -> Dict[str, Dict]:
        # PPO 的 critic 与 actor 同属一个 ActorCriticPolicy，已经随 agent_net 一并保存，无独立离线网络。
        return {}

    def update(self, batch_data: EpisodeBatch[SingleAgentFields]) -> Dict[str, Any]:
        if isinstance(batch_data, EpisodeBatch):
            batch = batch_data
            batch_hash = batch.metadata["batch_hash"]
        else:
            batch_hash = transition_batch_hash(batch_data)
            batch = SAEpisodeTransition.to_batch_tensor(episode_trans=batch_data, device=self.device)
        if batch_hash == self._last_batch_hash:
            return {
                "skip_update": True,
                "skip_reason": "same_sample_hash",
            }
        self._last_batch_hash = batch_hash
        self._assert_batch_finite(batch)
        self._prepare_rollout_buffer(batch)
        policy_state = {
            key: value.detach().cpu().clone()
            for key, value in self.ppo.policy.state_dict().items()
        }
        self._assert_policy_finite("before PPO.train")
        try:
            self.ppo.train()
            self._assert_policy_finite("after PPO.train")
        except Exception as exc:
            if not self._policy_is_finite():
                self.ppo.policy.load_state_dict(policy_state)
                self.ppo.policy.to(self.device)
                raise RuntimeError(
                    "SB3PPOLearner detected non-finite PPO policy parameters during PPO.train; "
                    "the previous finite parameters have been restored. "
                    "Usually this means the PPO batch/window is too correlated, reward scale is too large, "
                    "or the device backend is numerically unstable."
                ) from exc
            raise
        self.update_cnt += 1
        return self._collect_stats()

    def dynamic_modify_config(self, updates: Dict[str, Any]) -> None:
        """
        按学习令牌动态同步SB3 PPO算法超参。

        SB3 PPO部分字段在对象构造时固化，这里同步PPO.train会读取的运行期字段；
        结构型字段（如policy/hidden_dim）不在运行中重建网络。
        """
        super().dynamic_modify_config(updates)
        algo_configs = self.config.algo_config
        self.ppo.learning_rate = algo_configs.lr
        self.ppo.lr_schedule = FloatSchedule(algo_configs.lr)
        self.ppo.batch_size = algo_configs.minibatch_size
        self.ppo.n_epochs = algo_configs.n_epochs
        self.ppo.clip_range = FloatSchedule(algo_configs.clip_range)
        self.ppo.clip_range_vf = (
            None if algo_configs.clip_range_vf is None else FloatSchedule(algo_configs.clip_range_vf)
        )
        self.ppo.normalize_advantage = algo_configs.normalize_advantage
        self.ppo.ent_coef = algo_configs.ent_coef
        self.ppo.vf_coef = algo_configs.vf_coef
        self.ppo.max_grad_norm = algo_configs.max_grad_norm
        self.ppo.target_kl = algo_configs.target_kl


    def _build_ppo(self, learner_config: LearnerConfig) -> PPO:
        algo_config = learner_config.algo_config
        assert isinstance(algo_config, PpoConfig), (
            f"SB3PPOLearner requires PpoConfig, given={type(algo_config).__name__}"
        )
        space_env = _SpaceEnv(self.agent.obs_dim, self.agent.action_dim)
        policy_kwargs = {
            "net_arch": {
                "pi": [algo_config.hidden_dim, algo_config.hidden_dim],
                "vf": [algo_config.hidden_dim, algo_config.hidden_dim],
            }
        }
        ppo = PPO(
            policy=algo_config.policy,
            env=space_env,
            learning_rate=algo_config.lr,
            n_steps=algo_config.n_steps,
            batch_size=algo_config.minibatch_size,
            n_epochs=algo_config.n_epochs,
            gamma=learner_config.gamma,
            gae_lambda=algo_config.gae_lambda,
            clip_range=algo_config.clip_range,
            clip_range_vf=algo_config.clip_range_vf,
            normalize_advantage=algo_config.normalize_advantage,
            ent_coef=algo_config.ent_coef,
            vf_coef=algo_config.vf_coef,
            max_grad_norm=algo_config.max_grad_norm,
            target_kl=algo_config.target_kl,
            policy_kwargs=policy_kwargs,
            device=str(self.device),
            verbose=0,
        )
        # 离线训练不会调用 PPO.learn()，logger 不会被自动初始化；这里挂一个无输出 logger，
        # 让 PPO.train() 内部的 logger.record 正常写入 name_to_value，供 _collect_stats 读取。
        ppo.set_logger(Logger(folder=None, output_formats=[]))
        return ppo

    def _prepare_rollout_buffer(self, batch: SABatchTensorData) -> None:
        """用框架样本重建 SB3 RolloutBuffer：补算旧策略 value/log_prob 并做 per-episode GAE。"""
        algo_config = self.config.algo_config
        device = self.device
        num_transitions = batch.obs.shape[0]
        obs_dim = batch.obs.shape[1]

        obs = batch.obs.to(device).float()
        next_obs = batch.next_obs.to(device).float()
        actions_long = batch.actions.to(device).long().reshape(-1)

        self.ppo.policy.set_training_mode(False)
        with torch.no_grad():
            if batch.old_values is not None and batch.old_log_probs is not None:
                # 样本侧已记录行为策略的 value / log_prob，直接使用，得到严格 on-policy 的 PPO。
                values = batch.old_values.to(device).reshape(-1)
                log_probs = batch.old_log_probs.to(device).reshape(-1)
            else:
                # 样本未携带行为策略统计量时，用当前策略补算(first-epoch ratio≈1)，得到近似 on-policy。
                eval_values, log_probs, _ = self.ppo.policy.evaluate_actions(obs, actions_long)
                values = eval_values.reshape(-1)
                log_probs = log_probs.reshape(-1)
            next_values = self.ppo.policy.predict_values(next_obs).reshape(-1)

        values_np = values.detach().cpu().numpy()
        next_values_np = next_values.detach().cpu().numpy()
        rewards_np = batch.rewards.reshape(-1).cpu().numpy()
        terminated_np = batch.terminated.reshape(-1).cpu().numpy()
        episode_starts_np = batch.episode_starts.reshape(-1).cpu().numpy()

        if batch.advantages is not None and batch.returns is not None:
            advantages_np = batch.advantages.reshape(-1).cpu().numpy().astype(np.float32)
            returns_np = batch.returns.reshape(-1).cpu().numpy().astype(np.float32)
        else:
            advantages_np, returns_np = self._compute_gae(
                rewards=rewards_np,
                values=values_np,
                next_values=next_values_np,
                terminated=terminated_np,
                episode_starts=episode_starts_np,
                gamma=self.config.gamma,
                gae_lambda=algo_config.gae_lambda,
            )

        buffer = RolloutBuffer(
            buffer_size=num_transitions,
            observation_space=self.agent.obs_dim,
            action_space=self.agent.action_dim,
            device=device,
            gae_lambda=algo_config.gae_lambda,
            gamma=self.config.gamma,
            n_envs=1,
        )
        buffer.observations = obs.detach().cpu().numpy().reshape(
            num_transitions, 1, obs_dim
        ).astype(buffer.observation_space.dtype, copy=False)
        buffer.actions = batch.actions.detach().cpu().numpy().reshape(
            num_transitions, 1, buffer.action_dim
        ).astype(buffer.action_space.dtype, copy=False)
        buffer.values = values_np.reshape(num_transitions, 1)
        buffer.log_probs = log_probs.detach().cpu().numpy().reshape(num_transitions, 1)
        buffer.advantages = advantages_np.reshape(num_transitions, 1)
        buffer.returns = returns_np.reshape(num_transitions, 1)
        buffer.rewards = rewards_np.reshape(num_transitions, 1).astype(np.float32)
        buffer.episode_starts = episode_starts_np.reshape(num_transitions, 1).astype(np.float32)
        buffer.pos = num_transitions
        buffer.full = True
        buffer.generator_ready = False
        self.ppo.rollout_buffer = buffer

    @staticmethod
    def _compute_gae(
            rewards: np.ndarray,
            values: np.ndarray,
            next_values: np.ndarray,
            terminated: np.ndarray,
            episode_starts: np.ndarray,
            gamma: float,
            gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """逐 episode 计算 GAE(λ)；episode 末尾 terminated 不 bootstrap，截断/数据截止则用 V(s_{t+1}) bootstrap。"""
        num_transitions = len(rewards)
        advantages = np.zeros(num_transitions, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(num_transitions)):
            is_episode_end = (t == num_transitions - 1) or (episode_starts[t + 1] > 0.5)
            if is_episode_end:
                if terminated[t] > 0.5:
                    next_non_terminal = 0.0
                    next_value = 0.0
                else:
                    next_non_terminal = 1.0
                    next_value = float(next_values[t])
                # episode 边界处重置 GAE 累积量，避免跨 episode 泄漏。
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                last_gae = delta
            else:
                next_non_terminal = 1.0
                next_value = float(values[t + 1])
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values.astype(np.float32)
        return advantages, returns.astype(np.float32)

    def _collect_stats(self) -> Dict[str, Any]:
        """读取 SB3 PPO.train() 写入 logger 的训练指标。"""
        from scheduler.model import normalize_sb3_learn_metrics
        stats = normalize_sb3_learn_metrics(
            {key: value for key, value in self.ppo.logger.name_to_value.items()}
        )
        stats["update_cnt"] = self.update_cnt
        return stats

    @staticmethod
    def _assert_batch_finite(batch: SABatchTensorData) -> None:
        """训练前检查样本张量，避免把脏样本送进SB3后才在分布构造处报NaN。"""
        required_tensors = {
            "obs": batch.obs,
            "actions": batch.actions,
            "last_actions": batch.last_actions,
            "rewards": batch.rewards,
            "next_obs": batch.next_obs,
            "terminated": batch.terminated,
            "truncated": batch.truncated,
            "episode_starts": batch.episode_starts,
        }
        optional_tensors = {
            "old_log_probs": batch.old_log_probs,
            "old_values": batch.old_values,
            "returns": batch.returns,
            "advantages": batch.advantages,
        }
        for name, tensor in required_tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"SB3PPOLearner received non-finite batch tensor: {name}")
        for name, tensor in optional_tensors.items():
            if tensor is not None and not torch.isfinite(tensor).all():
                raise ValueError(f"SB3PPOLearner received non-finite batch tensor: {name}")

    def _policy_is_finite(self) -> bool:
        """检查当前PPO策略网络参数是否仍为有限值。"""
        return all(torch.isfinite(param).all().item() for param in self.ppo.policy.parameters())

    def _assert_policy_finite(self, stage: str) -> None:
        """训练前后检查策略网络参数，快速定位NaN出现阶段。"""
        if not self._policy_is_finite():
            raise RuntimeError(f"SB3PPOLearner policy contains non-finite parameters {stage}.")

    def _validate_agent(self, learner_config: LearnerConfig) -> None:
        if self.agent.config.agent_output_type != "pi_logits":
            raise ValueError("SB3PPOLearner requires agent_output_type='pi_logits'")
        # PPO 是单智能体算法：绑定的 Agent 必须只管理一个实体(agent_list 长度为 1)。
        if self.agent.n_agents != 1:
            raise ValueError(
                f"SB3PPOLearner is single-agent only, expected agent.n_agents == 1, "
                f"got {self.agent.n_agents} (agent_list={self.agent.agent_list})"
            )

    def _assert_policy(self) -> None:
        assert isinstance(self.ppo, PPO), f"self.ppo must be PPO, given={type(self._ppo).__name__}"
        assert isinstance(self.ppo.policy, ActorCriticPolicy), (
            f"self.ppo.policy must be ActorCriticPolicy, given={type(self.ppo.policy).__name__}"
        )
