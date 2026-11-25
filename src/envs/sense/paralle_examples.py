"""不依赖外部仿真的Parallel同步环境示例，用于二次开发参考和回归测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np

from config import EnvConfig
from envs.action.model import Action
from envs.base import AgentName, ParallelAgentOrderContext, ParallelEnvironment
from envs.state.base import State


@dataclass
class ParallelCounterState(State):
    """最小Parallel示例的联合全局状态。"""

    # 各环境AgentID当前的累积计数值。
    values: dict[str, float] = field(default_factory=dict)


class _ParallelCounterSimulation:
    """仅在内存中同步执行联合动作的最小Simulation实现。"""

    def __init__(self, agent_ids: list[str]):
        # 仿真维护的固定AgentID顺序。
        self.agent_ids = list(agent_ids)
        # 当前联合态势，每个Agent对应一个计数器。
        self.state = ParallelCounterState(
            entities=[],
            values={agent_id: 0.0 for agent_id in self.agent_ids},
        )
        # 最近一次send_actions接收的各AgentID计数增量。
        self.last_deltas: dict[str, float] = {}

    def reset(self) -> ParallelCounterState:
        """把全部计数器和联合动作缓存恢复为初值。"""
        self.state = ParallelCounterState(
            entities=[],
            values={agent_id: 0.0 for agent_id in self.agent_ids},
        )
        self.last_deltas = {}
        return self.state

    @staticmethod
    def _action_delta(action) -> float:
        """把框架Action或原始离散动作转换为计数器增量。"""
        if hasattr(action, "parameters"):
            return float(action.parameters.get("delta", 1.0))
        return 1.0 if int(action) == 1 else -1.0

    def send_actions(self, actions) -> None:
        """一次接收本物理步全部活动AgentID的联合动作。"""
        if set(actions) != set(self.agent_ids):
            raise ValueError(
                "Parallel counter simulation requires one action for every AgentID; "
                f"expected={self.agent_ids}, given={list(actions)}"
            )
        self.last_deltas = {
            agent_id: self._action_delta(action)
            for agent_id, action in actions.items()
        }

    def forward_step(self, interval_secs: float) -> ParallelCounterState:
        """在同一次物理转移中同步应用联合动作的所有增量。"""
        values = dict(self.state.values)
        for agent_id, delta in self.last_deltas.items():
            values[agent_id] += delta
        self.state = ParallelCounterState(
            entities=[],
            sim_time=self.state.sim_time + float(interval_secs),
            values=values,
        )
        return self.state

    def close(self) -> None:
        """内存仿真没有外部连接需要释放。"""
        return None


class CounterParallelEnvironment(ParallelEnvironment):
    """所有Agent先生成动作，再在一个物理步中同步更新计数器的Parallel示例。"""

    # 框架Parallel示例的环境标识和渲染能力声明。
    metadata = {"name": "dodo_counter_parallel_v0", "render_modes": []}

    def __init__(
            self,
            agent_ids: list[str],
            max_steps: int = 8,
            policy_agent_names: Optional[list[str]] = None,
    ):
        config = EnvConfig(
            agent_names=list(agent_ids),
            max_step_per_episode=max_steps,
            step_interval_secs=1.0,
            state_dim=len(agent_ids),
            use_sim=False,
        )
        super().__init__(config=config, connect_sim=False)
        # 用内存联合计数器替代外部Simulation，使示例可独立运行。
        self.sim = _ParallelCounterSimulation(list(agent_ids))
        # 内存仿真实现了Simulation协议，因此启用基类标准reset/step流程。
        self.connect_sim = True
        # Episode Action需要控制Policy组顺序时，由环境业务配置提供可返回的AgentName。
        self.policy_agent_names = tuple(policy_agent_names or agent_ids)
        # 每个Agent有减一和加一两个离散动作。
        self.action_dim = {
            agent_id: gym.spaces.Discrete(2)
            for agent_id in agent_ids
        }
        # 每个Agent都观测同一份联合计数器状态。
        self.obs_dim = {
            agent_id: gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(len(agent_ids),),
                dtype=np.float32,
            )
            for agent_id in agent_ids
        }

    def state(self) -> np.ndarray:
        """按possible_agents固定槽位顺序输出联合全局状态。"""
        return np.asarray(
            [self.post_state.values[agent_id] for agent_id in self.possible_agents],
            dtype=np.float32,
        )

    def get_obs(self) -> dict[str, np.ndarray]:
        """返回各AgentID的当前观测；示例使用完全可观测设置。"""
        state = self.state()
        return {
            agent_id: state.copy()
            for agent_id in self.possible_agents
        }

    def get_avail_agent_actions(self, agent_name: str, init: bool = False) -> list[int]:
        """两个计数动作在任意状态下都可用。"""
        return [1, 1]

    def get_reward(self, **kwargs) -> dict[str, float]:
        """各Agent的即时奖励等于其在本联合步中的计数增量。"""
        return {
            agent_id: float(self.sim.last_deltas.get(agent_id, 0.0))
            for agent_id in self.possible_agents
        }

    def done(self) -> tuple[bool, bool, bool]:
        """示例没有业务终止条件，仅在达到物理步上限时截断。"""
        return False, self.step_num >= self.max_step_per_episode, False

    def episode_agent_order_resolve(
        self,
        episode_agent_action: Action,
        context: ParallelAgentOrderContext,
    ) -> AgentName:
        """演示Parallel环境如何解释Episode Agent的完整动作。"""
        agent_order_rule = episode_agent_action.agent_order_rule
        pending_agent_names = tuple(
            name for name in self.policy_agent_names
            if name not in context.resolved_agent_names
        )
        if agent_order_rule in {"env_native", "declared_order"}:
            return pending_agent_names[0]
        if agent_order_rule == "reverse_policy_groups":
            return pending_agent_names[-1]
        raise ValueError(
            f"Unsupported counter Parallel agent_order_rule={agent_order_rule!r}."
        )


class SingleAgentCounterParallelEnvironment(CounterParallelEnvironment):
    """只有一个环境AgentID的Parallel基类接入示例。"""

    def __init__(self, max_steps: int = 8):
        super().__init__(["agent_0"], max_steps=max_steps)


class MultiAgentCounterParallelEnvironment(CounterParallelEnvironment):
    """三个Agent在每个物理步中同步执行联合动作的示例。"""

    def __init__(self, max_steps: int = 8):
        super().__init__(["agent_0", "agent_1", "agent_2"], max_steps=max_steps)
