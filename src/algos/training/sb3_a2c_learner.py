from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from stable_baselines3 import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import FloatSchedule

from algos.agents import Agent
from algos.map import AlogsMap
from algos.learners import Learner
from config import A2CConfig, LearnerConfig
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
    """只向 SB3 提供观测/动作空间；框架采样流程不会调用该环境的 step/reset。"""

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


# SB3A2CLearner：单智能体 A2C 学习器，基于 stable_baselines3.A2C。
# 框架 ReplayBuffer 仍以 episode 为单位存储，update 时拍平为单智能体 transition，
# 按 episode 边界计算 GAE/return，再交给 SB3 A2C 执行一次整批 actor-critic 更新。
@learner_plugin(A2CConfig)
class SB3A2CLearner(Learner[SingleAgentFields]):
    """DODORL Learner 到 stable_baselines3 A2C 的单智能体适配层。"""
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
        supported_agent_types = AlogsMap.compatible_agent_types(AlogsMap.a2c)
        assert agent.agent_type in supported_agent_types, (
            f"SB3A2CLearner supports agent types {supported_agent_types}, "
            f"given={agent.agent_type}, learner={config.name}"
        )
        self._a2c: Optional[A2C] = None
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
    def a2c(self) -> A2C:
        assert self._a2c is not None, "SB3A2CLearner.a2c must be initialized before access."
        return self._a2c

    @property
    def agent_net(self) -> nn.Module:
        return self.a2c.policy

    @property
    def models(self) -> Dict[str, nn.Module]:
        # SB3 ActorCriticPolicy 同时包含 actor 和 critic，作为一个完整模型发布。
        return {"agent_net": self.a2c.policy}

    def snapshot_state(self) -> Dict[str, Any]:
        """
        保存 A2C 全部训练状态：ActorCriticPolicy state_dict（含 actor/critic）、
        优化器 state_dict、update 计数器、防重复批次哈希、SB3 内部更新步数与时间步。
        """
        return {
            "policy_state": self.a2c.policy.state_dict(),
            "optimizer_state": self.a2c.policy.optimizer.state_dict(),
            "update_cnt": self.update_cnt,
            "_last_batch_hash": self._last_batch_hash,
            "a2c_n_updates": self.a2c._n_updates,
            "a2c_num_timesteps": self.a2c.num_timesteps,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.a2c.policy.load_state_dict(state["policy_state"])
        self.a2c.policy.optimizer.load_state_dict(state["optimizer_state"])
        self.update_cnt = state["update_cnt"]
        self._last_batch_hash = state["_last_batch_hash"]
        self.a2c._n_updates = state["a2c_n_updates"]
        self.a2c.num_timesteps = state["a2c_num_timesteps"]

    def build_empty_network(self, learner_config: LearnerConfig, **kwargs) -> Tuple[nn.Module, ...]:
        self._validate_agent(learner_config)
        self._a2c = self._build_a2c(learner_config)
        self._assert_policy()
        return (self.a2c.policy,)

    def build_network_from_checkpoint(
            self,
            checkpoint: Dict[str, NetWeight],
            learner_config: LearnerConfig,
            **kwargs,
    ) -> Tuple[nn.Module, ...]:
        self.build_empty_network(learner_config, **kwargs)
        self.a2c.policy.load_state_dict(extract_state_dict(checkpoint["agent_net"]), strict=False)
        self._assert_policy()
        return (self.a2c.policy,)

    def _offline_net_weights(self) -> Dict[str, Dict]:
        # critic 和 actor 共同位于 ActorCriticPolicy，无需额外导出离线网络。
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
        self._prepare_rollout_buffer(batch)
        self.a2c.train()
        self.update_cnt += 1
        return self._collect_stats()

    def dynamic_modify_config(self, updates: Dict[str, Any]) -> None:
        """按学习令牌同步 A2C 超参；网络结构参数不允许运行期变更。"""
        if not updates:
            return
        current = self.config.algo_config
        assert isinstance(current, A2CConfig), (
            f"SB3A2CLearner.config.algo_config must be A2CConfig, given={type(current).__name__}"
        )
        if "policy" in updates and updates["policy"] != current.policy:
            raise ValueError("A2C policy cannot be modified without rebuilding the network.")
        if "hidden_dim" in updates and updates["hidden_dim"] != current.hidden_dim:
            raise ValueError("A2C hidden_dim cannot be modified without rebuilding the network.")
        optimizer_changed = (
            ("use_rms_prop" in updates and updates["use_rms_prop"] != current.use_rms_prop)
            or ("rms_prop_eps" in updates and updates["rms_prop_eps"] != current.rms_prop_eps)
        )
        super().dynamic_modify_config(updates)
        algo_configs = self.config.algo_config
        self.a2c.learning_rate = algo_configs.lr
        self.a2c.lr_schedule = FloatSchedule(algo_configs.lr)
        self.a2c.n_steps = algo_configs.n_steps
        self.a2c.gae_lambda = algo_configs.gae_lambda
        self.a2c.ent_coef = algo_configs.ent_coef
        self.a2c.vf_coef = algo_configs.vf_coef
        self.a2c.max_grad_norm = algo_configs.max_grad_norm
        self.a2c.normalize_advantage = algo_configs.normalize_advantage

        if optimizer_changed:
            self._rebuild_optimizer(algo_configs)
        else:
            for param_group in self.a2c.policy.optimizer.param_groups:
                param_group["lr"] = algo_configs.lr
                if algo_configs.use_rms_prop:
                    param_group["eps"] = algo_configs.rms_prop_eps


    def _build_a2c(self, learner_config: LearnerConfig) -> A2C:
        algo_config = learner_config.algo_config
        assert isinstance(algo_config, A2CConfig), (
            f"SB3A2CLearner requires A2CConfig, given={type(algo_config).__name__}"
        )
        space_env = _SpaceEnv(self.agent.obs_dim, self.agent.action_dim)
        policy_kwargs = {
            "net_arch": {
                "pi": [algo_config.hidden_dim, algo_config.hidden_dim],
                "vf": [algo_config.hidden_dim, algo_config.hidden_dim],
            }
        }
        a2c = A2C(
            policy=algo_config.policy,
            env=space_env,
            learning_rate=algo_config.lr,
            n_steps=algo_config.n_steps,
            gamma=learner_config.gamma,
            gae_lambda=algo_config.gae_lambda,
            ent_coef=algo_config.ent_coef,
            vf_coef=algo_config.vf_coef,
            max_grad_norm=algo_config.max_grad_norm,
            rms_prop_eps=algo_config.rms_prop_eps,
            use_rms_prop=algo_config.use_rms_prop,
            normalize_advantage=algo_config.normalize_advantage,
            policy_kwargs=policy_kwargs,
            device=str(self.device),
            verbose=0,
        )
        # 本适配层不调用 A2C.learn()，因此手动挂载无输出 logger 收集 train() 指标。
        a2c.set_logger(Logger(folder=None, output_formats=[]))
        return a2c

    def _prepare_rollout_buffer(self, batch: SABatchTensorData) -> None:
        """用框架样本重建 SB3 RolloutBuffer，按 episode 分隔计算 GAE。"""
        algo_config = self.config.algo_config
        assert isinstance(algo_config, A2CConfig)
        device = self.device
        num_transitions = batch.obs.shape[0]
        obs_dim = batch.obs.shape[1]

        obs = batch.obs.to(device).float()
        next_obs = batch.next_obs.to(device).float()
        actions_long = batch.actions.to(device).long().reshape(-1)

        self.a2c.policy.set_training_mode(False)
        with torch.no_grad():
            if batch.old_values is not None:
                # 有采样时价值估计时优先使用，保持严格 on-policy 语义。
                values = batch.old_values.to(device).reshape(-1)
            else:
                values = self.a2c.policy.predict_values(obs).reshape(-1)
            _, log_probs, _ = self.a2c.policy.evaluate_actions(obs, actions_long)
            next_values = self.a2c.policy.predict_values(next_obs).reshape(-1)

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
        self.a2c.rollout_buffer = buffer

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
        """逐 episode 计算 GAE；terminated 不 bootstrap，truncated/样本截止使用 V(s_{t+1})。"""
        num_transitions = len(rewards)
        advantages = np.zeros(num_transitions, dtype=np.float32)
        last_gae = 0.0
        for step in reversed(range(num_transitions)):
            is_episode_end = (
                step == num_transitions - 1 or episode_starts[step + 1] > 0.5
            )
            if is_episode_end:
                if terminated[step] > 0.5:
                    next_non_terminal = 0.0
                    next_value = 0.0
                else:
                    next_non_terminal = 1.0
                    next_value = float(next_values[step])
                delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
                last_gae = delta
            else:
                next_value = float(values[step + 1])
                delta = rewards[step] + gamma * next_value - values[step]
                last_gae = delta + gamma * gae_lambda * last_gae
            advantages[step] = last_gae
        returns = advantages + values.astype(np.float32)
        return advantages, returns.astype(np.float32)

    def _rebuild_optimizer(self, algo_config: A2CConfig) -> None:
        """仅在令牌修改优化器类型/数值时重建，其他动态参数不丢失优化器状态。"""
        if algo_config.use_rms_prop:
            optimizer_class = torch.optim.RMSprop
            optimizer_kwargs = {
                "alpha": 0.99,
                "eps": algo_config.rms_prop_eps,
                "weight_decay": 0,
            }
        else:
            optimizer_class = torch.optim.Adam
            optimizer_kwargs = {"eps": 1e-5}
        self.a2c.policy.optimizer_class = optimizer_class
        self.a2c.policy.optimizer_kwargs = optimizer_kwargs
        self.a2c.policy.optimizer = optimizer_class(
            self.a2c.policy.parameters(),
            lr=algo_config.lr,
            **optimizer_kwargs,
        )

    def _collect_stats(self) -> Dict[str, Any]:
        """读取 A2C.train() 写入 logger 的训练指标。"""
        from scheduler.model import normalize_sb3_learn_metrics
        stats = normalize_sb3_learn_metrics(
            {key: value for key, value in self.a2c.logger.name_to_value.items()}
        )
        stats["update_cnt"] = self.update_cnt
        return stats

    def _validate_agent(self, learner_config: LearnerConfig) -> None:
        if not isinstance(learner_config.algo_config, A2CConfig):
            raise TypeError(
                f"SB3A2CLearner requires A2CConfig, "
                f"given={type(learner_config.algo_config).__name__}"
            )
        if self.agent.config.agent_output_type != "pi_logits":
            raise ValueError("SB3A2CLearner requires agent_output_type='pi_logits'")
        if self.agent.n_agents != 1:
            raise ValueError(
                f"SB3A2CLearner is single-agent only, expected agent.n_agents == 1, "
                f"got {self.agent.n_agents} (agent_list={self.agent.agent_list})"
            )

    def _assert_policy(self) -> None:
        assert isinstance(self.a2c, A2C), f"self.a2c must be A2C, given={type(self._a2c).__name__}"
        assert isinstance(self.a2c.policy, ActorCriticPolicy), (
            f"self.a2c.policy must be ActorCriticPolicy, given={type(self.a2c.policy).__name__}"
        )
