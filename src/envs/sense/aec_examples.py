"""不依赖外部仿真的AEC示例环境，用于二次开发参考和API回归测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np

from config import EnvConfig
from envs.action.model import Action
from envs.base import AECAgentOrderContext, AECEnvironment
from envs.state.base import State


@dataclass
class CounterState(State):
    """最小AEC示例的全局状态。"""

    # 各环境AgentID当前的累积计数值。
    values: dict[str, float] = field(default_factory=dict)


class _CounterSimulation:
    def __init__(self, agent_ids: list[str]):
        # 示例仿真需要维护的固定AgentID顺序。
        self.agent_ids = agent_ids
        # 当前仿真态势，每个Agent对应一个计数器。
        self.state = CounterState(entities=[], values={agent: 0.0 for agent in agent_ids})
        # 最近一次send_actions选中的AEC行动者。
        self.last_actor: Optional[str] = None
        # 最近一次动作对计数器产生的增量。
        self.last_delta = 0.0

    def reset(self):
        self.state = CounterState(entities=[], values={agent: 0.0 for agent in self.agent_ids})
        self.last_actor = None
        self.last_delta = 0.0
        return self.state

    def send_actions(self, actions):
        self.last_actor, action = next(iter(actions.items()))
        if hasattr(action, "parameters"):
            self.last_delta = float(action.parameters.get("delta", 1.0))
        else:
            self.last_delta = 1.0 if int(action) == 1 else -1.0

    def forward_step(self, interval_secs):
        values = dict(self.state.values)
        values[self.last_actor] += self.last_delta
        self.state = CounterState(
            entities=[],
            sim_time=self.state.sim_time + float(interval_secs),
            values=values,
        )
        return self.state

    def close(self):
        return None


class CounterAECEnvironment(AECEnvironment):
    """每个Agent动作都会立刻改变自身计数器的标准AEC示例。"""

    # PettingZoo API测试使用的示例环境标识和渲染能力声明。
    metadata = {"name": "dodo_counter_aec_v0", "render_modes": []}

    def __init__(self, agent_ids: list[str], max_steps: int = 8):
        config = EnvConfig(
            agent_names=list(agent_ids),
            max_step_per_episode=max_steps,
            step_interval_secs=1.0,
            state_dim=len(agent_ids),
            use_sim=False,
        )
        super().__init__(config=config, connect_sim=False)
        # 用内存计数器仿真替代外部Simulation，使示例可独立运行。
        self.sim = _CounterSimulation(list(agent_ids))
        # 内存仿真同样实现了Simulation协议，因此启用标准reset/step流程。
        self.connect_sim = True
        # 每个Agent有减一和加一两个离散动作。
        self.action_dim = {agent: gym.spaces.Discrete(2) for agent in agent_ids}
        # 每个Agent都观测全部计数器，便于展示AEC步进而非部分可观测建模。
        self.obs_dim = {
            agent: gym.spaces.Box(-np.inf, np.inf, shape=(len(agent_ids),), dtype=np.float32)
            for agent in agent_ids
        }

    def state(self) -> np.ndarray:
        return np.asarray(
            [self.post_state.values[agent] for agent in self.possible_agents],
            dtype=np.float32,
        )

    def get_obs(self) -> dict[str, np.ndarray]:
        state = self.state()
        return {agent: state.copy() for agent in self.possible_agents}

    def get_avail_agent_actions(self, agent_name: str, init: bool = False) -> list[int]:
        return [1, 1]

    def get_reward(self, **kwargs) -> dict[str, float]:
        return {
            agent: float(self.sim.last_delta if agent == self.sim.last_actor else 0.0)
            for agent in self.possible_agents
        }

    def done(self) -> tuple[bool, bool, bool]:
        return False, self.step_num >= self.max_step_per_episode, False

    def episode_agent_order_resolve(
        self,
        episode_agent_action: Action,
        context: AECAgentOrderContext,
    ) -> str:
        """演示同一AEC环境如何解释Episode Agent的完整动作。"""
        agent_order_rule = episode_agent_action.agent_order_rule
        if agent_order_rule == "env_native":
            return self.agent_selection
        if agent_order_rule == "first_live":
            return self.agents[0]
        if agent_order_rule == "reverse_round_robin":
            order = tuple(reversed(self.agents))
            return order[len(context.resolved_agent_ids) % len(order)]
        raise ValueError(f"Unsupported counter AEC agent_order_rule={agent_order_rule!r}.")


class SingleAgentCounterAECEnvironment(CounterAECEnvironment):
    def __init__(self, max_steps: int = 8):
        super().__init__(["agent_0"], max_steps=max_steps)


class MultiAgentCounterAECEnvironment(CounterAECEnvironment):
    def __init__(self, max_steps: int = 8):
        super().__init__(["agent_0", "agent_1", "agent_2"], max_steps=max_steps)
