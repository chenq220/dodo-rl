from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy

from algos.agents import Agent, record_action_warp
from algos.agents.input_shape import AgentInputShape
from common.plugins import agent_plugin
from config import SB3AgentConfig

if TYPE_CHECKING:
    from common.types import (
        ActionMask,
        AgentActon,
        AgentHiddenIn,
        AgentHiddenOut,
        AgentID,
        AgentLastAction,
        ObsType,
    )


# SB3Agent：执行侧承载 stable_baselines3 的单智能体策略网络。
# PPO/A2C 使用 ActorCriticPolicy，DQN 使用 DQNPolicy；三者共用同一个业务 Agent 类型。
# 严格遵守框架约定：self.agent_list 始终是 list，单智能体下长度恒为 1，构建时强校验。
@agent_plugin(SB3AgentConfig, default_obs_agent_id=False)
class SB3Agent(Agent):
    """
    项目 Agent 接口到 stable_baselines3 单智能体策略网络的适配层。

    只负责执行期动作计算：PPO/A2C 从策略分布采样动作，DQN 从 Q 值做 epsilon-greedy；
    最终都映射回业务 Action 实体。MLP 无循环结构，隐藏态为占位。
    """

    # ActorCriticPolicy 构造需要一个 lr_schedule；Agent 执行侧不优化，给常数占位即可。
    _DUMMY_LR = 3e-4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_on_policy_stats: Optional[tuple[float, float]] = None

    def consume_on_policy_stats(self) -> tuple[Optional[float], Optional[float]]:
        """取出上一步 pi_logits 策略采样时记录的 (log_prob, value)。"""
        stats = self._last_on_policy_stats
        self._last_on_policy_stats = None
        if stats is None:
            return None, None
        return stats[0], stats[1]

    def reset_net(self, model: nn.Module):
        """SB3 执行侧只同步权重，不 deepcopy Learner 的优化器和计算图状态。"""
        if not isinstance(model, (ActorCriticPolicy, DQNPolicy)):
            raise TypeError(
                f"reset_net model type mismatch, expected=ActorCriticPolicy|DQNPolicy, "
                f"given={type(model).__name__}"
            )
        if not isinstance(self.agent_net, type(model)):
            raise TypeError(
                f"reset_net model type mismatch, current={type(self.agent_net).__name__}, "
                f"given={type(model).__name__}"
            )
        self.agent_net.load_state_dict(model.state_dict(), strict=False)
        self.agent_net.to(self.device)
        self.agent_net.eval()

    def build_empty_network(self, dim_info: AgentInputShape, **kwargs) -> nn.Module:
        self._assert_single_agent()
        return self._build_policy()

    def build_network_from_checkpoint(
            self, checkpoint, dim_info: AgentInputShape, **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        self._assert_single_agent()
        policy = self._build_policy()
        state_dict = self._extract_state_dict(checkpoint)
        policy.load_state_dict(state_dict, strict=False)
        return policy, policy.state_dict()

    def get_init_hidden_state(self) -> Dict["AgentID", Any]:
        # SB3 ActorCriticPolicy(MLP) 无循环状态，返回占位以满足框架隐藏态接口。
        return {agent_id: None for agent_id in self.agent_list}

    @record_action_warp
    def get_actions(
            self,
            obs: Dict["AgentID", "ObsType"],
            agent_list: List["AgentID"],
            hidden_in: "AgentHiddenIn",
            action_mask: "ActionMask" = None,
            epsilon: float = 0.0,
            evaluate: bool = False,
            last_actions: "AgentLastAction" = None,
            temperature: float = 1.0,
            deterministic: bool = False,
            **kwargs
    ) -> Tuple["AgentActon", "AgentHiddenOut"]:
        if len(agent_list) != 1:
            raise ValueError(
                f"{type(self).__name__} is single-agent only, expected one agent per step, got {agent_list}"
            )
        agent_id = agent_list[0]
        self._last_on_policy_stats = None

        if self.config.agent_output_type == "pi_logits":
            action_idx, log_prob, value = self._pi_action_with_stats(
                obs_value=obs[agent_id],
                agent_id=agent_id,
                action_mask=action_mask,
                evaluate=evaluate,
                deterministic=deterministic,
            )
            self._last_on_policy_stats = (log_prob, value)
        elif evaluate or deterministic:
            action_idx = self._greedy_action(obs[agent_id], agent_id, action_mask)
        elif self.config.agent_output_type == "q":
            action_idx = self._epsilon_greedy_action(obs[agent_id], agent_id, action_mask, epsilon)
        else:
            raise ValueError(f"Unsupported SB3 agent_output_type: {self.config.agent_output_type}")

        actions = {agent_id: self.action_table.action_map[action_idx]}
        return actions, self.get_init_hidden_state()

    def _pi_action_with_stats(
            self,
            obs_value: "ObsType",
            agent_id: "AgentID",
            action_mask: "ActionMask",
            evaluate: bool,
            deterministic: bool,
    ) -> tuple[int, float, float]:
        """采样/贪心动作，并记录行为策略 log_prob 与 value 供 PPO/A2C on-policy 学习。"""
        if not isinstance(self.agent_net, ActorCriticPolicy):
            raise TypeError(
                f"_pi_action_with_stats requires ActorCriticPolicy, given={type(self.agent_net).__name__}"
            )
        if evaluate or deterministic:
            action_idx = self._greedy_action(obs_value, agent_id, action_mask)
        else:
            probs = self._action_probs(obs_value, agent_id, action_mask)
            action_idx = int(np.random.choice(len(probs), p=probs))

        obs_array = np.asarray(obs_value, dtype=np.float32).flatten()
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor([action_idx], device=self.device, dtype=torch.long)

        self.agent_net.eval()
        with torch.no_grad():
            values, log_probs, _ = self.agent_net.evaluate_actions(obs_tensor, action_tensor)
        return (
            action_idx,
            float(log_probs.reshape(-1)[0].item()),
            float(values.reshape(-1)[0].item()),
        )

    def _greedy_action(
            self,
            obs_value: "ObsType",
            agent_id: "AgentID",
            action_mask: "ActionMask",
    ) -> int:
        if self.config.agent_output_type == "q":
            q_values = self._q_values(obs_value, agent_id, action_mask)
            return int(torch.argmax(q_values).item())
        probs = self._action_probs(obs_value, agent_id, action_mask)
        return int(np.argmax(probs))

    def _epsilon_greedy_action(
            self,
            obs_value: "ObsType",
            agent_id: "AgentID",
            action_mask: "ActionMask",
            epsilon: float,
    ) -> int:
        valid_indices = self._valid_action_indices(agent_id, action_mask)
        if np.random.random() < epsilon:
            return int(np.random.choice(valid_indices))
        q_values = self._q_values(obs_value, agent_id, action_mask)
        return int(torch.argmax(q_values).item())

    def _action_probs(
            self,
            obs_value: "ObsType",
            agent_id: "AgentID",
            action_mask: "ActionMask",
    ) -> np.ndarray:
        """前向 SB3 策略网络得到(屏蔽非法动作并归一化后的)动作概率分布。"""
        if not isinstance(self.agent_net, ActorCriticPolicy):
            raise TypeError(
                f"_action_probs requires ActorCriticPolicy, given={type(self.agent_net).__name__}"
            )
        obs_array = np.asarray(obs_value, dtype=np.float32).flatten()
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.agent_net.eval()
        with torch.no_grad():
            distribution = self.agent_net.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.squeeze(0)

        probs = self._mask_and_normalize(probs, agent_id, action_mask)
        return probs.detach().cpu().numpy()

    def _q_values(
            self,
            obs_value: "ObsType",
            agent_id: "AgentID",
            action_mask: "ActionMask",
    ) -> torch.Tensor:
        """前向 SB3 DQNPolicy 得到 Q 值，并把非法动作置为极小值。"""
        if not isinstance(self.agent_net, DQNPolicy):
            raise TypeError(f"_q_values requires DQNPolicy, given={type(self.agent_net).__name__}")
        obs_array = np.asarray(obs_value, dtype=np.float32).flatten()
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.agent_net.eval()
        with torch.no_grad():
            q_values = self.agent_net.q_net(obs_tensor).squeeze(0)

        return self._mask_q_values(q_values, agent_id, action_mask)

    def _mask_q_values(
            self,
            q_values: torch.Tensor,
            agent_id: "AgentID",
            action_mask: "ActionMask",
    ) -> torch.Tensor:
        if action_mask is None or agent_id not in action_mask or action_mask[agent_id] is None:
            return q_values
        mask = action_mask[agent_id]
        if len(mask) != self.action_table.n_actions:
            raise ValueError(
                f"action_mask for {agent_id} has dim {len(mask)}, expected {self.action_table.n_actions}"
            )
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if not mask_tensor.any():
            raise ValueError("DQN q-values have no valid action after masking")
        return q_values.masked_fill(~mask_tensor, torch.finfo(q_values.dtype).min)

    def _valid_action_indices(
            self,
            agent_id: "AgentID",
            action_mask: "ActionMask",
    ) -> np.ndarray:
        if action_mask is None or agent_id not in action_mask or action_mask[agent_id] is None:
            return np.arange(self.action_table.n_actions)
        mask = np.asarray(action_mask[agent_id], dtype=bool)
        if len(mask) != self.action_table.n_actions:
            raise ValueError(
                f"action_mask for {agent_id} has dim {len(mask)}, expected {self.action_table.n_actions}"
            )
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            raise ValueError("DQN epsilon-greedy has no valid action after masking")
        return valid_indices

    def _mask_and_normalize(
            self,
            probs: torch.Tensor,
            agent_id: "AgentID",
            action_mask: "ActionMask",
    ) -> torch.Tensor:
        if action_mask is not None and agent_id in action_mask and action_mask[agent_id] is not None:
            mask = action_mask[agent_id]
            if len(mask) != self.action_table.n_actions:
                raise ValueError(
                    f"action_mask for {agent_id} has dim {len(mask)}, expected {self.action_table.n_actions}"
                )
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            probs = probs.masked_fill(~mask_tensor, 0.0)

        total = probs.sum()
        if total <= 0:
            raise ValueError("policy distribution has no valid action after masking")
        return probs / total

    def _build_policy(self) -> ActorCriticPolicy:
        """构建与配对 SB3 Learner 一致的策略网络，保证权重可互相同步。"""
        if self.config.agent_output_type == "q":
            return self._build_dqn_policy()
        if self.config.agent_output_type != "pi_logits":
            raise ValueError(f"Unsupported SB3 agent_output_type: {self.config.agent_output_type}")
        return self._build_actor_critic_policy()

    def _build_actor_critic_policy(self) -> ActorCriticPolicy:
        net_arch = {
            "pi": [self.config.hidden_dim, self.config.hidden_dim],
            "vf": [self.config.hidden_dim, self.config.hidden_dim],
        }
        policy = ActorCriticPolicy(
            observation_space=self.obs_dim,
            action_space=self.action_dim,
            lr_schedule=lambda _: self._DUMMY_LR,
            net_arch=net_arch,
        )
        return policy.to(self.device)

    def _build_dqn_policy(self) -> DQNPolicy:
        policy = DQNPolicy(
            observation_space=self.obs_dim,
            action_space=self.action_dim,
            lr_schedule=lambda _: self._DUMMY_LR,
            net_arch=[self.config.hidden_dim, self.config.hidden_dim],
        )
        return policy.to(self.device)

    @staticmethod
    def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()
        if isinstance(checkpoint, dict):
            for key in ("agent_net", "model_state_dict", "state_dict"):
                inner = checkpoint.get(key)
                if isinstance(inner, dict):
                    return inner
            if all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
        raise ValueError("unsupported checkpoint format; expected module or state_dict")

    def _assert_single_agent(self) -> None:
        # 框架约定 agent_list 必须是 list；SB3Agent 当前只支持单智能体。
        if not isinstance(self.agent_list, list):
            raise TypeError(f"agent_list must be a list, got {type(self.agent_list)}")
        if len(self.agent_list) != 1:
            raise ValueError(
                f"{type(self).__name__} is single-agent only, expected len(agent_list) == 1, "
                f"got {len(self.agent_list)} (agent_list={self.agent_list})"
            )
