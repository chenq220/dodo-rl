from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class AgentInputShape:
    """
    Agent 网络输入维度描述，统一替代旧版 agent_dim。

    value 属性表示单个 agent 在一次前向推理中的拼接输入维度；
    to_scheme 为循环执行控制器提供输入维度描述。
    """
    # 单个 agent 的私有观测向量维度。
    obs_dim: int
    # 离散动作空间大小，即 ActionTable.n_actions。
    action_dim: int
    # 当前策略网络服务的智能体数量。
    n_agents: int
    # 是否把上一动作 one-hot 拼接到当前观测输入中。
    obs_last_action: bool
    # 是否把 agent id one-hot 拼接到当前观测输入中。
    obs_agent_id: bool
    # RNN/GRU 隐藏状态维度；非循环网络可保留为 0。
    hidden_dim: int = 0
    # 循环网络层数；当前 RNNAgent 通常使用 1。
    num_layers: int = 1

    # Complete typed layout; action_dim retains its legacy categorical meaning.
    action_layout: dict | None = None

    @property
    def value(self) -> int:
        """返回单个 agent 实际输入网络的拼接维度。"""
        input_shape = self.obs_dim
        if self.obs_last_action:
            input_shape += (self.action_layout["previous_action_dim"]
                            if self.action_layout is not None else self.action_dim)
        if self.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def to_scheme(self, state_shape: Any = None) -> Dict[str, Dict[str, Any]]:
        """转换为离散动作循环执行控制器需要的输入维度。"""
        if self.action_layout is not None and self.action_layout["kind"] != "discrete":
            raise ValueError("Recurrent Q controller currently supports only discrete action spaces")
        scheme = {
            "obs": {"vshape": self.obs_dim},
            "actions_onehot": {"vshape": (self.action_dim,)},
        }
        if state_shape is not None:
            scheme["state"] = {"vshape": state_shape}
        return scheme
