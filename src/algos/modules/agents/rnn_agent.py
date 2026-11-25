from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, Generic, TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from algos.agents import Agent, record_action_warp
from algos.modules.agents.recurrent_controller import RecurrentController
from algos.agents.input_shape import AgentInputShape
from common.types import AgentID, ObsType
from common.plugins import agent_plugin
from config import RnnAgentConfig
from utils.checkpoint import validated_checkpoint_state_dict

if TYPE_CHECKING:
    from common.types import (
        ActionMask,
        AgentActon,
        AgentHiddenIn,
        AgentHiddenOut,
        AgentLastAction
    )


class _OneStepBatch:
    def __init__(self, data, device):
        self.data = data
        self.device = device
        self.batch_size = 1
        self.max_seq_length = 2

    def __getitem__(self, key):
        return self.data[key]


# 共享网络RNN Agent：所有业务智能体共用同一套循环网络参数，并可通过agent_id输入区分身份。
# 适用于同构协同任务，例如多个同构智能体动作空间、观测结构和角色能力一致的场景。
@agent_plugin(RnnAgentConfig, default_obs_agent_id=True)
class RNNAgent(Agent, Generic[AgentID, ObsType]):
    """
    项目 Agent 接口到原生循环网络和控制器的适配层。

    这个类只负责执行期动作计算：把项目的obs、last_action、hidden_state
    转成循环控制器能消费的一步输入，再把Q值转成项目要求的动作概率分布。
    """

    mac_cls = RecurrentController
    epymarl_sharing_scope = "all"
    force_agent_id_input = None
    mac: RecurrentController

    def build_empty_network(self, dim_info: AgentInputShape, **kwargs) -> nn.Module:
        self.mac = self._build_mac(dim_info)
        return self.mac.agent

    def build_network_from_checkpoint(self, checkpoint, dim_info, **kwargs) -> Tuple[
        nn.Module, Dict[str, torch.Tensor]]:
        self.mac = self._build_mac(dim_info)
        state_dict = validated_checkpoint_state_dict(
            self.mac.agent, checkpoint, agent_name=self.name, model_name="agent_net",
            expected_dimensions={
                "n_agents": dim_info.n_agents,
                "obs_dim": dim_info.obs_dim,
                "n_actions": dim_info.action_dim,
                "state_shape": self.config.state_shape,
            },
        )
        self.mac.agent.load_state_dict(state_dict)
        return self.mac.agent, self.mac.agent.state_dict()

    def reset_net(self, model: nn.Module):
        super().reset_net(model)
        self.mac.agent = self.agent_net

    def get_init_hidden_state(self) -> Dict["AgentID", Any]:
        init_hidden = self.mac.agent.init_hidden().to(self.device)
        if init_hidden.shape[0] == self.n_agents:
            return {agent_id: init_hidden[idx:idx + 1].clone() for idx, agent_id in enumerate(self.agent_list)}
        return {agent_id: init_hidden.clone() for agent_id in self.agent_list}

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
        action_probs, hidden_out = self._calc_action_probs(
            obs=obs,
            agent_list=agent_list,
            hidden_in=hidden_in,
            action_mask=action_mask,
            last_actions=last_actions,
            temperature=temperature,
            **kwargs,
        )
        actions = {}
        # epsilon-greedy只适用于Q值输出；策略输出(pi_logits)的探索由策略分布采样完成。
        effective_epsilon = epsilon if self.config.agent_output_type == "q" else 0.0
        for agent_id in agent_list:
            probs = self._normalize_action_probs(action_probs[agent_id])
            if evaluate or deterministic:
                action_idx = int(np.argmax(probs))
            elif effective_epsilon > 0 and random.random() < effective_epsilon:
                action_idx = self._sample_available_action(agent_id, probs, action_mask)
            else:
                action_idx = int(np.random.choice(len(probs), p=probs))
            actions[agent_id] = self.action_table.action_map[action_idx]

        return actions, hidden_out

    def _calc_action_probs(
            self,
            obs: Dict["AgentID", "ObsType"],
            agent_list: List["AgentID"],
            hidden_in: "AgentHiddenIn",
            action_mask: "ActionMask",
            last_actions: "AgentLastAction" = None,
            temperature: float = 1.0,
            **kwargs
    ) -> Tuple[Dict["AgentID", np.ndarray], "AgentHiddenOut"]:
        """将网络输出转换为当前 Agent 本次动作选择需要的概率分布。"""
        self.mac.agent.eval()
        batch = self._build_action_batch(obs, agent_list, last_actions, action_mask)
        self.mac.hidden_states = self._hidden_to_tensor(hidden_in, agent_list).unsqueeze(0)

        with torch.no_grad():
            agent_outputs = self.mac.forward(batch, t=1)

        agent_outputs = agent_outputs.squeeze(0)
        if self.config.agent_output_type == "q":
            agent_outputs = self._mask_unavailable_actions(agent_outputs, agent_list, action_mask)
            probs = self._q_values_to_greedy_probs(agent_outputs)
        elif self.config.agent_output_type == "pi_logits":
            probs = self._mask_and_normalize_policy(
                agent_outputs, agent_list, action_mask, temperature=temperature,
            )
        else:
            raise ValueError(f"Unsupported agent_output_type: {self.config.agent_output_type}")
        hidden_out = self.mac.hidden_states.squeeze(0).detach()

        return (
            {
                agent_id: probs[self.agent_list.index(agent_id)].detach().cpu().numpy()
                for agent_id in agent_list
            },
            {
                agent_id: hidden_out[self.agent_list.index(agent_id):self.agent_list.index(agent_id) + 1]
                for agent_id in agent_list
            },
        )

    def _sample_available_action(
            self,
            agent_id: "AgentID",
            probs: np.ndarray,
            action_mask: "ActionMask",
    ) -> int:
        if action_mask and agent_id in action_mask and action_mask[agent_id] is not None:
            avail_actions_idx = [i for i, avail in enumerate(action_mask[agent_id]) if avail == 1]
            return int(random.choice(avail_actions_idx)) if avail_actions_idx else int(np.argmax(probs))
        return int(random.randint(0, len(probs) - 1))

    def _build_mac(self, dim_info: AgentInputShape) -> RecurrentController:
        # 执行期不需要全局state，state_shape给占位值即可；训练期由Learner按环境真实state构建。
        self.config.algo_config.sharing_scope = self.epymarl_sharing_scope
        if self.force_agent_id_input is not None:
            self.config.obs_agent_id = self.force_agent_id_input
        self.config.bind_runtime(
            n_agents=self.n_agents,
            n_actions=dim_info.action_dim,
            state_shape=1,
            device=self.device,
        )
        mac_input_shape = AgentInputShape(
            obs_dim=dim_info.obs_dim,
            action_dim=dim_info.action_dim,
            n_agents=self.n_agents,
            obs_last_action=self.config.obs_last_action,
            obs_agent_id=self.config.obs_agent_id,
            hidden_dim=dim_info.hidden_dim,
            num_layers=dim_info.num_layers,
        )
        self.dim_info = mac_input_shape
        return self.mac_cls(
            scheme=mac_input_shape.to_scheme(),
            groups={"agents": self.n_agents},
            args=self.config,
        )

    def _build_action_batch(
            self,
            obs: Dict["AgentID", "ObsType"],
            agent_list: List["AgentID"],
            last_actions: "AgentLastAction",
            action_mask: "ActionMask",
    ) -> _OneStepBatch:
        obs_tensor = torch.zeros(1, 2, self.n_agents, self.dim_info.obs_dim, dtype=torch.float32, device=self.device)
        actions_onehot = torch.zeros(
            1, 2, self.n_agents, self.dim_info.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        avail_actions = torch.ones_like(actions_onehot)
        # IDEA： 这里只要自己的态势，如果考虑agent上下游编排从这里入手
        for agent_id in agent_list:
            agent_idx = self.agent_list.index(agent_id)
            value = obs[agent_id]
            obs_tensor[0, 1, agent_idx] = torch.as_tensor(value, dtype=torch.float32, device=self.device).flatten()

            if last_actions is not None:
                action = last_actions.get(agent_id)
                if action is not None:
                    action_idx = int(action.action_idx if hasattr(action, "action_idx") else action)
                    if 0 <= action_idx < self.dim_info.action_dim:
                        actions_onehot[0, 0, agent_idx, action_idx] = 1.0

            if action_mask is not None:
                mask = action_mask.get(agent_id)
                if mask is not None:
                    if len(mask) != self.dim_info.action_dim:
                        raise ValueError(
                            f"action_mask for {agent_id} has dim {len(mask)}, "
                            f"expected {self.dim_info.action_dim}"
                        )
                    avail_actions[0, 1, agent_idx] = torch.as_tensor(
                        mask,
                        dtype=torch.float32,
                        device=self.device,
                    )

        return _OneStepBatch(
            data={
                "obs": obs_tensor,
                "actions_onehot": actions_onehot,
                "avail_actions": avail_actions,
            },
            device=self.device,
        )

    def _hidden_to_tensor(self, hidden_in: "AgentHiddenIn", agent_list: List["AgentID"]) -> torch.Tensor:
        hidden = torch.zeros(self.n_agents, self.dim_info.hidden_dim, dtype=torch.float32, device=self.device)
        for agent_id in agent_list:
            agent_idx = self.agent_list.index(agent_id)
            hidden[agent_idx] = torch.as_tensor(hidden_in[agent_id], dtype=torch.float32, device=self.device).reshape(
                -1)
        return hidden

    def _mask_unavailable_actions(
            self,
            q_values: torch.Tensor,
            agent_list: List["AgentID"],
            action_mask: "ActionMask",
    ) -> torch.Tensor:
        if action_mask is None:
            return q_values
        q_values = q_values.clone()
        for agent_id in agent_list:
            mask = action_mask.get(agent_id)
            if mask is None:
                continue
            agent_idx = self.agent_list.index(agent_id)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            q_values[agent_idx].masked_fill_(~mask_tensor, -float("inf"))
        return q_values

    def _mask_and_normalize_policy(
            self,
            policy_outputs: torch.Tensor,
            agent_list: List["AgentID"],
            action_mask: "ActionMask",
            temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        把策略输出转成合法概率分布。

        MAC 在 pi_logits 模式下已经 softmax，此处输入通常是概率(非负且和≈1)；
        对概率只做 0 屏蔽 + 重归一化。若输入仍是 logits(含负数)，再走 masked softmax。
        """
        outputs = policy_outputs.clone()
        row_sums = outputs.sum(dim=-1)
        is_prob = (
            torch.all(outputs >= 0)
            and torch.all(row_sums > 0)
            and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)
        )

        if is_prob:
            probs = outputs
            if action_mask is not None:
                for agent_id in agent_list:
                    mask = action_mask.get(agent_id)
                    if mask is None:
                        continue
                    agent_idx = self.agent_list.index(agent_id)
                    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                    probs[agent_idx] = probs[agent_idx].masked_fill(~mask_tensor, 0.0)
            sums = probs.sum(dim=-1, keepdim=True)
            if torch.any(sums <= 0):
                raise ValueError("policy distribution has no valid action after masking")
            return probs / sums

        logits = outputs
        if action_mask is not None:
            for agent_id in agent_list:
                mask = action_mask.get(agent_id)
                if mask is None:
                    continue
                agent_idx = self.agent_list.index(agent_id)
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                logits[agent_idx] = logits[agent_idx].masked_fill(~mask_tensor, -float("inf"))

        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _normalize_action_probs(probs: np.ndarray) -> np.ndarray:
        """采样前兜底：裁剪负值/NaN 并归一化，避免 np.random.choice 报错。"""
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            return np.ones_like(probs) / len(probs)
        return probs / total

    def _q_values_to_greedy_probs(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        项目接口要求返回动作概率分布；QMIX执行策略本质是Q值贪婪选择。

        epsilon探索由Agent基类统一处理，所以这里返回最高合法Q值对应的one-hot分布。
        """
        probs = []
        for row in q_values:
            if torch.isneginf(row).all():
                probs.append(torch.ones_like(row) / row.numel())
            else:
                prob = torch.zeros_like(row)
                prob[torch.argmax(row)] = 1.0
                probs.append(prob)
        return torch.stack(probs, dim=0)
