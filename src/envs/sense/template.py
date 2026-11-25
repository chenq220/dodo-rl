from __future__ import annotations

from typing import Dict, Generic, List

import gymnasium as gym
import numpy as np

from config import EnvConfig
from envs.base import AgentID, ActionType, ParallelEnvironment, EnvStepReturn, ObsType, StateType
from utils.tools import zeros_ndarray


class TemplateEnv(ParallelEnvironment, Generic[AgentID, ObsType, StateType]):
    """
    Parallel场景二次开发模板，开发前请仔细阅读 ParallelEnvironment 的属性和已实现方法。

    子类通常只需要补齐 state/get_obs/get_avail_agent_actions/done/get_reward。
    reset/step/send_actions/record_sample 等主流程已由 ParallelEnvironment 基类实现，除非业务场景
    有特殊时序要求，否则不要重写。
    """

    def __init__(self, config: EnvConfig, sim_config, connect_sim=True, sim_manager=None, role_id=None):
        super().__init__(
            config=config,
            sim_config=sim_config,
            connect_sim=connect_sim,
            sim_manager=sim_manager,
            role_id=role_id,
        )
        # >>> 自定义属性 <<<

    def state(self) -> np.ndarray:
        """ 全局态势 """
        state = self.post_state
        if state is None:
            return zeros_ndarray(self.state_space())
        # 获取全局态势逻辑
        global_state = []
        # >>> 编码实现 <<<
        expected_state_dim = self.state_space().shape[0]
        assert len(global_state) == expected_state_dim, (
            f"TemplateEnv.state global_state dimension mismatch: "
            f"actual={len(global_state)}, expected={expected_state_dim}. "
            f"Please fill global_state with exactly EnvConfig.state_dim features or override state_space()."
        )
        # 向量化
        state_vector = np.array(global_state, dtype=np.float32)
        norm = np.linalg.norm(state_vector)
        # 归一化
        if norm > 0:
            state_vector = state_vector / norm
        return state_vector

    def get_obs(self) -> dict[AgentID, ObsType]:
        """ 获取智能体观察态势 """
        agent_obs_dict = {}
        for agent in self.possible_agents:
            # 获取智能体观察态势逻辑
            agent_obs = []
            # >>> 编码实现 <<<
            expected_obs_dim = self.observation_space(agent).shape[0]
            assert len(agent_obs) == expected_obs_dim, (
                f"TemplateEnv.get_obs observation dimension mismatch: "
                f"agent={agent}, actual={len(agent_obs)}, expected={expected_obs_dim}. "
                f"Please fill agent_obs with exactly AgentConfig.obs_dim features."
            )
            # 向量化
            agent_obs_vector = np.array(agent_obs, dtype=np.float32)
            norm = np.linalg.norm(agent_obs_vector)
            # 归一化
            if norm > 0:
                agent_obs_vector_norm = agent_obs_vector / norm
            else:
                agent_obs_vector_norm = agent_obs_vector
            agent_obs_dict[agent] = agent_obs_vector_norm
        return agent_obs_dict

    def get_avail_agent_actions(self, agent_name: AgentID, init: bool = False) -> List[int]:
        """ 获取动作掩码 """
        agent_mask = [1] * self.action_space(agent_name).n
        return agent_mask

    def done(self) -> tuple[bool, bool, bool]:
        """ 步进结束 """
        terminated = False
        need_resume = True
        truncated = self.step_num >= self.max_step_per_episode
        return terminated, truncated, need_resume

    def get_reward(self, **kwargs) -> Dict[AgentID, float]:
        """ 获取奖励 """
        reward_dict = {}
        for agent in self.possible_agents:
            # 获取奖励逻辑
            reward = 0.0
            # >>> 编码实现 <<<
            reward_dict[agent] = reward
        return reward_dict

    def step(self, actions: Dict[AgentID, ActionType]) -> EnvStepReturn:
        """ 步进 """
        # 基类已实现，可重写
        return super().step(actions)

    def send_actions(self, actions: Dict[AgentID, ActionType]) -> Dict[AgentID, ActionType]:
        """ 打印日志 """
        # 基类已实现，可重写
        return super().send_actions(actions)

    def step_log(self):
        """ 打印步进日志 """
        # 基类已实现，可重写
        super().step_log()
