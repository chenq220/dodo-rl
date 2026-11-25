import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List, Optional

# 请根据你的实际路径调整导入
from algos.agents import Agent, record_action_warp
from algos.agents.input_shape import AgentInputShape
from common.types import AgentID, ObsType, AgentHiddenIn, AgentHiddenOut, ActionMask, AgentActon
from common.plugins import agent_plugin
from common.exception import EnvironmentException
from config import MonteCarloAgentConfig


@agent_plugin(MonteCarloAgentConfig, default_obs_agent_id=False)
class RandomAgent(Agent):
    """
    纯蒙特卡洛（Monte Carlo）随机采样智能体。
    无视温度、探索率（epsilon）和评估模式，完全基于动作掩码进行绝对公平的随机动作抽取。
    内部不包含任何可训练参数，也不要求配置同名Learner。
    """

    sig = 'monte_carlo'
    supported_action_spaces = ("discrete", "multi_discrete", "continuous", "multi_continuous")

    def build_empty_network(self, dim_info: AgentInputShape, **kwargs) -> nn.Module:
        """兼容显式请求网络构建的旧调用方。"""
        return nn.Identity()

    def build_network_from_checkpoint(self, checkpoint, dim_info: AgentInputShape, **kwargs) \
            -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """伪装加载：返回空网络和空字典"""
        return nn.Identity(), {}

    def get_init_hidden_state(self, agent_ids: Optional[List[AgentID]] = None) -> Dict[AgentID, Any]:
        """蒙特卡洛属于无记忆策略，不需要 RNN 隐藏状态，直接返回 None"""
        agent_ids = agent_ids or self.agent_list
        return {agent_id: None for agent_id in agent_ids}

    @record_action_warp
    def get_actions(self,
                    obs: Dict[AgentID, ObsType],
                    agent_list: List[AgentID],
                    hidden_in: AgentHiddenIn,
                    action_mask: ActionMask = None,
                    epsilon: float = 0.0,
                    evaluate: bool = False,
                    last_actions=None,
                    temperature: float = 1.0,
                    deterministic: bool = False,
                    **kwargs) -> Tuple[AgentActon, AgentHiddenOut]:
        """
        ⚠️ 核心逻辑拦截！
        无视外部传进来的 epsilon, evaluate, deterministic, temperature 等花里胡哨的参数。
        强制在合法的 Action Mask 范围内做 Python 原生的绝对纯净随机。
        """
        unknown = set(agent_list) - set(self.agent_list)
        if unknown:
            raise EnvironmentException(
                detail=f"蒙特卡洛智能体收到未管理的智能体ID：{sorted(unknown)}"
            )

        actions: AgentActon = {}
        hidden_out: AgentHiddenOut = {}

        for agent_id in agent_list:
            mask = action_mask.get(agent_id) if action_mask else None
            try:
                actions[agent_id] = self.action_space.sample(np.random, mask)
            except ValueError as exc:
                raise EnvironmentException(detail=f"Random action for {agent_id}: {exc}") from exc
            hidden_out[agent_id] = hidden_in.get(agent_id) if hidden_in else None

        return actions, hidden_out
